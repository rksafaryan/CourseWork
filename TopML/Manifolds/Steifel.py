import jax.numpy as jnp
from jax.scipy.linalg import sqrtm
from jax.scipy.linalg import qr
import jax
from jax import jit


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

    def random_point(self, num_points=1):
        """
        Generate random points on the Stiefel manifold.

        Parameters:
        num_points (int): Number of points to generate.

        Returns:
        points (list): List of generated points on the Stiefel manifold.
        """
        points = []
        for _ in range(num_points):
            # Generate a random n x p matrix
            A = jax.random.uniform(jax.random.PRNGKey(0), shape=(self.n, self.p))

            # Perform QR decomposition to obtain a point on the Stiefel manifold
            Q, _ = qr(A, mode='economic')

            points.append(Q)

        return points

    def Projection(self, X, E):
        """
        Optimization Algorithms on Matrix Manifolds
        Projection from ambient space to tangent space at x
        X - point on a manifold
        E - vector from ambient space
        P_X E = (I-X^TX)E + X skew(X^TE)
        """
        return (jnp.eye(X.shape[0]) - X @ X.T) @ E + X @ (X.T @ E - E.T @ X) / 2

    def Retraction_polar(self, X, E):
        """
        Optimization Algorithms on Matrix Manifolds
        Projection from ambient space to tangent space at x
        X - point on a manifold
        E - vector from ambient space
        R_X(E) = (X+E)(I_p + E.TE)^(-1/2)
        """
        return (X + E) @ sqrtm(jnp.linalg.inv(jnp.eye(X.shape[0]) + E.T @ E))

    def Retraction_qr(self, X, E):
        Q, _ = jnp.linalg.qr(X + E)
        return Q

    def retraction_svd(self, X, E):
        u, _, vh = jnp.linalg.svd(X + E, full_matrices=False)
        return u @ vh

    def distance(self, X, Y, base=None):

        if base is None:
            base = Y

        return jnp.trace(jnp.eye(self.p) - X.T @ Y)

    def random_tangent_vector(self, point):
        vector = jax.random.normal(jax.random.PRNGKey(0), shape=point.shape)
        vector = self.Projection(point, vector)
        return vector / jnp.linalg.norm(vector)
