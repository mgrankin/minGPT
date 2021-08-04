"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import logging
logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import haiku as hk

from jax.experimental import optimizers
import optax
from optax import chain, clip_by_global_norm, scale_by_adam, scale, scale_by_schedule, add_decayed_weights
from jax import local_device_count
from jax.experimental.maps import xmap, mesh

from tqdm import tqdm
import math
import numpy as np
from typing import Mapping
import functools
from functools import partial
import pickle 

import torch
from torch.utils.data import Dataset, DataLoader

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    step_tokens = None # number of tokens predicted in one step, default = block_size
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    rng = jax.random.PRNGKey(42)

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def lr_schedule(config, step_items):
    def lr_sheduler(nstep):
        # decay the learning rate based on our progress
        n_tokens = jnp.array(nstep, float) * config.batch_size * step_items
        if config.lr_decay:
            progress = (n_tokens - config.warmup_tokens) / max(1, config.final_tokens - config.warmup_tokens)
            lr_mult = jnp.where(
                n_tokens < config.warmup_tokens, 
                    # linear warmup
                    n_tokens / jnp.fmax(1, config.warmup_tokens),
                    # cosine learning rate decay
                    jnp.fmax(0.1, 0.5 * (1.0 + jnp.cos(math.pi * progress))))
            lr = config.learning_rate * lr_mult
        else:
            lr = config.learning_rate
        return lr
    return lr_sheduler

def configure_decay_mask(params):
    """
    This function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    """
    # replace when registry is accessible
    # https://github.com/google/jax/blob/97a5719fcb40af7231b5f803f965063538282f8e/jax/_src/tree_util.py#L197
    tree_types = (tuple, list, dict, Mapping, type(None))
    
    def check_decay_list(key, parent_decays):
        if any([layer in key for layer in ['embeddings', 'layer_norm', 'multi_head_attention']]): return 0
        if key == 'b': return 0
        if 'linear' in key: return 1
        return parent_decays
    
    def check_decay(item, parent_decays):
        if not isinstance(item, tree_types): return parent_decays
        tree_type = type(item)
        if isinstance(item, (dict, Mapping)):
            tree = {k:check_decay(v, check_decay_list(k, parent_decays)) for k,v in item.items()}
        else:
            tree = [check_decay(v, parent_decays) for v in item]
        return tree_type(tree)
    
    mask = check_decay(params, -1)
    # validate that we considered every parameter
    assert all([decays >= 0 for decays in jax.tree_flatten(mask)[0]])
    return jax.tree_map(lambda x: x == 1, mask)

def trim_batch(batch):
    per_device_batch_size = batch.shape[0] // local_device_count()
    return batch[:per_device_batch_size * local_device_count()]

class Trainer:
    def __init__(self, hk_loss_fn, train_dataset, test_dataset, config):
        self.hk_loss_fn = hk_loss_fn
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

    def save_checkpoint(self, params, opt_state):
        if self.config.ckpt_path is None: return
        logger.info("saving to %s", self.config.ckpt_path )
        pickle.dump(params, open(self.config.ckpt_path + '/model.npy', "wb"))
        pickle.dump(opt_state, open(self.config.ckpt_path + '/optimizer.npy', "wb"))
    
    def init_params(self):
        self.config.rng, subkey = jax.random.split(self.config.rng)
        train_dl = DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        batch = next(iter(train_dl))
        xs, ys = map(jnp.array, batch)
        params = self.hk_loss_fn.init(subkey, xs[0], ys[0])
        logger.info("number of parameters: %d", sum([leave.size for leave in jax.tree_leaves(params)]))
        return params
            
    def train(self, params, opt_state=None):
        config = self.config
        lr_sheduler = lr_schedule(config, config.step_tokens if config.step_tokens is not None else self.train_dataset.block_size)
        
        optimiser = chain(
            clip_by_global_norm(config.grad_norm_clip),
            scale_by_adam(*config.betas),
            add_decayed_weights(config.weight_decay, configure_decay_mask(params)),
            scale_by_schedule(lr_sheduler),
            scale(-1),
        )
        if opt_state is None:
            opt_state = optimiser.init(params)
        loss_fn = self.hk_loss_fn.apply
        
        devices = jax.devices()        
        @partial(xmap, in_axes=[[...], [...], ['batch', ...], ['batch', ...], [...]], out_axes=[...], axis_resources={'batch': 'b'})
        def update(params, subkey, x, y, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params, subkey, x, y)
            
            grads = jax.lax.pmean(grads, axis_name='batch')
            loss = jax.lax.pmean(loss, axis_name='batch')
            
            updates, opt_state = optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state
                
        @partial(xmap, in_axes=[[...], [...], ['batch', ...], ['batch', ...]], out_axes=[...], axis_resources={'batch': 'b'})
        def get_loss(params, subkey, xs, ys):
            loss = loss_fn(params, subkey, xs, ys)
            return jax.lax.pmean(loss, axis_name='batch')
            
        def run_epoch(params, opt_state, it, split):
            is_train = split == 'train'
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(loader) if is_train else loader
            for batch in pbar:
                xs, ys = map(trim_batch, map(jnp.array, batch))
                # different rng on each device
                config.rng, subkey = jax.random.split(config.rng)
                
                # forward the model
                if is_train:
                    loss, params, opt_state = update(params, subkey, xs, ys, opt_state)
                else:
                    loss = get_loss(params, subkey, xs, ys)
                    
                losses.append(loss)
                
                if is_train:
                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss:.5f}. lr {lr_sheduler(it):e}")
                it += 1
                
            if not is_train:
                test_loss = float(jnp.mean(jnp.array(losses)))
                logger.info(f"test loss: {test_loss}")
                return test_loss
            
            return params, opt_state, it
        
        best_loss = float('inf')
        it = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            with mesh(devices, ('b')): 
                params, opt_state, it = run_epoch(params, opt_state, it, 'train')
                if self.test_dataset is not None:
                    test_loss = run_epoch(params, opt_state, 0, 'test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(params, opt_state)
                
        return params, opt_state
    