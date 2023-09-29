import jax.numpy as jnp
from jax.scipy.linalg import sqrtm
from jax.scipy.linalg import qr
import jax
from jax import jit
from jax.scipy.linalg import expm


class Steifel:
    def __init__(self, n, p):
        """
        Initialize the Stiefel manifold.

        Parameters:
        n (int): Dimension of the ambient space.
        k (int): Dimension of the Stiefel manifold.
        """
        self.n = n
        self.p = p

    def random_point(self, key=0, num_points=1):
        """
        Generate random points on the Stiefel manifold.

        Parameters:
        num_points (int): Number of points to generate.

        Returns:
        points (list): List of generated points on the Stiefel manifold.
        """
        points = []
        rng = jax.random.PRNGKey(key)
        for i in range(num_points):
            rng, _ = jax.random.split(rng)
            # Generate a random n x p matrix
            A = jax.random.uniform(rng, shape=(self.n, self.p))

            # Perform QR decomposition to obtain a point on the Stiefel manifold
            Q, _ = qr(A, mode='economic')

            points.append(Q)

        return jnp.squeeze(jnp.array(points))

    def exp(self, pt, v):
        A = pt.T @ v
        B = v @ pt.T
        exp = jax.linalg.expm(B - B.T) @ pt @ jax.linalg.expm(-A)
        return exp

    def projection(self, pt, v):
        """
        Optimization Algorithms on Matrix Manifolds
        Projection from ambient space to tangent space at x
        X - point on a manifold
        E - vector from ambient space
        P_X E = (I-X^TX)E + X skew(X^TE)
        """
        return (jnp.eye(pt.shape[0]) - pt @ pt.T) @ v + pt @ (pt.T @ v - v.T @ pt) / 2

    def retraction_polar(self, pt, v):
        """
        Optimization Algorithms on Matrix Manifolds
        Projection from ambient space to tangent space at x
        X - point on a manifold
        E - vector from ambient space
        R_X(E) = (X+E)(I_p + E.TE)^(-1/2)
        """
        return (pt + v) @ sqrtm(jnp.linalg.inv(jnp.eye(pt.shape[1]) + v.T @ v))

    def retraction_qr(self, pt, v):
        Q, _ = jnp.linalg.qr(pt + v)
        return Q

    def retraction_svd(self, pt, v):
        u, _, vh = jnp.linalg.svd(pt + v, full_matrices=False)
        return u @ vh

    def dist(self, X, Y):
        return jnp.trace(jnp.eye(self.p) - X.T @ Y)

    def random_tangent_vector(self, point):
        vector = jax.random.normal(jax.random.PRNGKey(0), shape=point.shape)
        vector = self.projection(point, vector)
        return vector / jnp.linalg.norm(vector)
