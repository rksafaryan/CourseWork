from jax import numpy as jnp
from jax import vmap, lax, jit


def pairwise_distance(pt_a, pt_b_set, manifold, test=False):
    """
        Compute the pairwise distance between a point and a set of points on the manifold.

        Parameters:
        pt_a (array): Single point (n x p matrix).
        pt_b_set (array): Set of points (batch_size x n x p matrix).
    """
    # Calculate the pairwise distances using vmap
    distances = vmap(manifold.dist, in_axes=(None, 0))(pt_a, pt_b_set)
    if test:
        print(distances)
    return jnp.mean(distances)
