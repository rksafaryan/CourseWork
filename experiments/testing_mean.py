from TopML.Manifolds.Grassmann import GrassmannCanonical
from TopML.Manifolds.utils import pairwise_distance
from jax import numpy as jnp
import jax
from jax.scipy.linalg import sqrtm
from jax.scipy.linalg import qr
from jax import jit
from functools import partial
from jax import vmap, lax, jit
from jax import grad, jacobian, custom_vjp
import numpy as np

N = 10
gr = GrassmannCanonical(5, 2)

point = gr.random_point(1)
set_of_points = gr.random_point(4, N)
print(pairwise_distance(point, set_of_points, gr))

dy = grad(pairwise_distance, argnums=0)
# distances = vmap(pairwise_distance, (0, None, None), 0)(set_of_points, set_of_points, gr)

for i in range(100):
    print(pairwise_distance(point, set_of_points, gr))
    anti_gradient = gr.egrad_to_rgrad(point, -dy(point, set_of_points, gr))
    point = gr.retr(point, anti_gradient)
"""
print(dy(point, set_of_points, gr))
dy_projected = lambda x, X: gr.project(x, dy(x, X, gr))
dy_projected_trace = lambda x, X: jnp.trace(gr.project(x, dy(x, X, gr)))

d2yy = jnp.squeeze(jacobian(dy_projected_trace, argnums=0)(point, set_of_points))
print(d2yy)"""
