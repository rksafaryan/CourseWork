from jax import numpy as jnp, vmap
import jax


class HypersphereCanonicalMetric:
    def __init__(self, m):
        self.m = m
        self.radius = 1

    def random_point(self, key, num_points=1):
        if num_points == 1:
            x = jax.random.normal(key, shape=(self.m,))
            norm_x = x / jnp.linalg.norm(x)
            return norm_x
        points = []
        if num_points > 1:
            for _ in range(num_points):
                key, _ = jax.random.split(key)
                x = jax.random.normal(key, shape=(self.m,))
                norm_x = x / jnp.linalg.norm(x)
                points.append(norm_x)
        return jnp.stack(points)

    def random_point1(self, key, num_points=1):
        """
        Generate random points on the surface of the hypersphere using JAX.
        """
        # Generate random spherical coordinates
        theta = jax.random.uniform(key, minval=0, maxval=2 * jnp.pi, shape=(num_points, self.m-1))
        phi = jnp.arccos(2 * jax.random.uniform(key, shape=(num_points, self.m-1)) - 1)

        # Convert spherical coordinates to Cartesian coordinates
        # For n-dimensional hypersphere, we use (n-1) spherical angles
        spherical_coords = jnp.column_stack((phi, theta))

        cartesian_coords = self.radius * jnp.hstack((
            jnp.prod(jnp.sin(spherical_coords[:, :-1]), axis=1, keepdims=True),
            jnp.cos(spherical_coords[:, -1:])
        ))

        # Stack the Cartesian coordinates with zeros to match the dimension
        if self.m > 3:
            zeros = jnp.zeros((num_points, self.m - 3))
            cartesian_coords = jnp.hstack((cartesian_coords, zeros))

        return cartesian_coords

    def exp(self, base_point, tangent_vec):
        norm = jnp.linalg.norm(tangent_vec)
        return base_point * jnp.cos(norm) + (tangent_vec / norm) * jnp.sin(norm)

    def log(self, base_point, point):
        coeff = self.dist(base_point, point)
        v = point.value - base_point.value
        proj = v - jnp.inner(v, base_point.value) * base_point.value
        log = coeff * (proj / jnp.linalg.norm(proj))
        return log

    def parallel_transport(self, start_point, end_point, tangent_vec):
        v = self.log(start_point, tangent_vec)
        v_norm = jnp.norm(v)
        inp = jnp.inner(v, tangent_vec)
        a = ((jnp.cos(v_norm) - 1) * inp) / v_norm
        b = (jnp.sin(v_norm) * inp) / v_norm
        pt = tangent_vec + a * tangent_vec - b * start_point
        return pt

    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):
        return jnp.inner(tangent_vec_a, tangent_vec_b)

    def dist(self, point_a, point_b):
        dist = jnp.arccos(jnp.clip(jnp.inner(point_a, point_b), -1, 1))
        return dist

    def projection(self, point, vector):
        """
        Projection from ambient space to tangent space at x
        point - point on a sphere
        vector - vector from ambient space
        """

        def sphere_proj(s, x):
            # Get normal vector as radius
            n = s / jnp.linalg.norm(s)
            # Find projection
            return x - jnp.dot(x - s, n) * n

        if len(vector.shape) > 1:
            return vmap(sphere_proj)(point, vector)
        else:
            return sphere_proj(point, vector)

    def retraction(self, point, vector):
        """
        Central projection on sphere surface
        point - point on a sphere
        vector - vector from tangent space at x
        """

        def sphere_retr(x, a):
            # step forward
            p = x + a
            return p * (jnp.linalg.norm(x) / jnp.linalg.norm(p))

        if len(point.shape) > 1:
            return vmap(sphere_retr)(point, vector)
        else:
            return sphere_retr(point, vector)

    @staticmethod
    def egrad_to_rgrad(base_point, egrad):
        return egrad.value - jnp.inner(base_point.value, egrad.value) * base_point.value