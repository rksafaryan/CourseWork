from jax import numpy as jnp
import jax


class GrassmannCanonical:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.dim = n * p - p * p
        self.shape = (n, p)
        self.ref_point = jnp.eye(n, p)

    def exp(self, b_pt, tv):
        u, s, vt = jnp.linalg.svd(tv, full_matrices=False)
        exp = (
                b_pt @ (vt.T * jnp.cos(s).reshape(1, -1)) @ vt
                + (u * jnp.sin(s).reshape(1, -1)) @ vt
        )
        return exp

    def retr(self, b_pt, tv):
        u, _, vt = jnp.linalg.svd(b_pt + tv, full_matrices=False)
        return u @ vt

    def log(self, b_pt, pt):
        ytx = pt.T @ b_pt
        At = pt.T - ytx @ b_pt.T
        Bt = jnp.linalg.solve(ytx, At)
        u, s, vt = jnp.linalg.svd(Bt.T, full_matrices=False)
        log = (u * jnp.arctan(s).reshape(1, -1)) @ vt
        return log

    def norm(self, b_pt, vec):
        return jnp.linalg.norm(vec, axis=(-1, -2))

    def dist(self, pt_a, pt_b):
        s = jnp.clip(jnp.linalg.svd(pt_a.T @ pt_b, compute_uv=False), a_max=1.0)
        dist = jnp.linalg.norm(jnp.arccos(s))
        return dist

    def inner_product(self, b_pt, tangent_vec_a, tangent_vec_b):
        ip = jnp.tensordot(tangent_vec_a, tangent_vec_b, axes=2)
        return ip

    def parallel_transport(self, s_pt, e_pt, tv):
        direction = self.log(s_pt, e_pt)
        u, s, vt = jnp.linalg.svd(direction, full_matrices=False)
        ut_delta = u.T @ tv
        pt = (
                (
                        s_pt @ (vt.T * -1 * jnp.sin(s).reshape(1, -1))
                        + (u * jnp.cos(s).reshape(1, -1))
                )
                @ ut_delta
                + tv
                - u @ ut_delta
        )
        return pt

    def egrad_to_rgrad(self, b_pt, egrad):
        return self.project(b_pt, egrad)

    def project(self, b_pt, vec):
        return vec - b_pt @ (b_pt.T @ vec)

    def random_point(self, key, num_points=1):
        if num_points == 1:
            q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(key), shape=(self.n, self.p)))
            return q
        else:
            # Generate a set of num_points random points
            points = []
            for k in range(num_points):
                q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(k*key), shape=(self.n, self.p)))
                points.append(q)
            return jnp.stack(points)


    def random_tangent_vector(self, point):
        tangent_vector = jax.random.normal(jax.random.PRNGKey(0), shape=point.shape)
        tangent_vector = self.project(point, tangent_vector)
        return tangent_vector / jnp.linalg.norm(tangent_vector)
