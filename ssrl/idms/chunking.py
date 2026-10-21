"""Bound temporary activations in row-independent IDM computations."""

import jax
import jax.numpy as jnp
import ninjax as nj


def chunked_rows(fn, inputs, chunk_size):
    """Map a state-reading Ninjax function over rows, preserving their order.

    Only the small, reduced outputs are stacked. Rematerializing each chunk
    prevents reverse-mode autodiff from retaining all chunks' activations.
    Padding repeats the last valid row and is removed from every output.
    Random samples must be drawn before this function to preserve their mapping
    to rows independently of the chunk size.
    """
    leaves = jax.tree.leaves(inputs)
    rows = leaves[0].shape[0]
    size = int(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if any(x.shape[0] != rows for x in leaves):
        raise ValueError("chunked inputs must have the same number of rows")
    if rows <= size:
        return fn(inputs)

    def step(carry, start):
        indices = jnp.minimum(start + jnp.arange(size), rows - 1)
        chunk = jax.tree.map(lambda x: x[indices], inputs)
        return carry, fn(chunk)

    _, outputs = nj.scan(
        nj.checkpoint(step), None, jnp.arange(0, rows, size), unroll=1
    )
    return jax.tree.map(
        lambda x: x.reshape((-1, *x.shape[2:]))[:rows], outputs
    )
