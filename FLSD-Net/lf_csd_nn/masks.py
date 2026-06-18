"""可微柔性頻帶遮罩。"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def generate_soft_masks(
    n: int, low_bound_idx: float, high_bound_idx: float, smoothness: float = 2.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = jnp.arange(n, dtype=jnp.float64)
    soft_low = jax.nn.sigmoid(smoothness * (idx - low_bound_idx)) * jax.nn.sigmoid(
        smoothness * (high_bound_idx - idx)
    )
    return soft_low, jnp.ones(n)
