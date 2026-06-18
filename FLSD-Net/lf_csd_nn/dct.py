"""正交 DCT-II 矩陣與 Hz ↔ bin 映射。"""

from __future__ import annotations

import jax.numpy as jnp


def generate_dct_matrix(n: int) -> jnp.ndarray:
    t = jnp.arange(n)
    k = jnp.arange(n)[:, jnp.newaxis]
    dct_mat = jnp.cos(jnp.pi * k * (2 * t + 1) / (2 * n))
    scale = jnp.ones(n) * jnp.sqrt(2.0 / n)
    scale = scale.at[0].set(jnp.sqrt(1.0 / n))
    return dct_mat * scale[:, jnp.newaxis]


def hz_to_dct_index(f_hz: float, n: int, fs: float) -> float:
    return float(f_hz) * 2.0 * n / fs


def dct_band_from_hz(
    n: int, fs: float, f_low: float, f_high: float, *, exclude_dc: bool = True
) -> tuple[float, float]:
    low = hz_to_dct_index(f_low, n, fs)
    high = hz_to_dct_index(f_high, n, fs)
    if exclude_dc:
        low = max(low, 1.0)
    low = max(low, 0.5)
    high = max(high, low + 1.0)
    return low, min(high, n - 1)
