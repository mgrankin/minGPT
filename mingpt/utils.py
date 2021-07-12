import jax
import jax.numpy as jnp
import haiku as hk

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from haiku import multinomial

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = jax.lax.top_k(logits, k)
    return jnp.where(logits < v[-1], -float('Inf'), logits)

def sample(params, model, config, x, steps, temperature=1.0, sample=False, top_k=None, rng=None, progress=False):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    if rng is None: 
        rng = jax.random.PRNGKey(random.randrange(2**31)) 
    block_size = config.block_size
    for k in tqdm(range(steps)) if progress else range(steps):
        x_cond = x if x.size <= block_size else x[-block_size:] # crop context if needed
        logits = model(params, jnp.array(x_cond))
        # pluck the logits at the final step and scale by temperature
        logits = logits[-1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply log_softmax to convert to log of probabilities (for hk.multinomial)
        probs = jax.nn.log_softmax(logits, axis=-1)
        # sample from the distribution or take the most likely
        if sample:
            rng, subkey = jax.random.split(rng)
            ix = hk.multinomial(subkey, probs, num_samples=1)
        else:
            _, ix = jax.lax.top_k(probs, k=1)
        # append to the sequence and continue
        x = jnp.concatenate((x, ix), axis=0)
    return x