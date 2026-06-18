"""可微 Park 變換：abc → dq。"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit


@jit
def abc_to_dq(v_abc: jnp.ndarray, theta: float | jnp.ndarray) -> jnp.ndarray:
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    cos_t_minus = jnp.cos(theta - 2 * jnp.pi / 3)
    cos_t_plus = jnp.cos(theta + 2 * jnp.pi / 3)
    sin_t_minus = jnp.sin(theta - 2 * jnp.pi / 3)
    sin_t_plus = jnp.sin(theta + 2 * jnp.pi / 3)
    matrix = (
        jnp.array(
            [
                [cos_t, cos_t_minus, cos_t_plus],
                [-sin_t, -sin_t_minus, -sin_t_plus],
                [1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 1 / jnp.sqrt(2)],
            ]
        )
        * jnp.sqrt(2 / 3)
    )
    return (matrix @ v_abc)[:2]
