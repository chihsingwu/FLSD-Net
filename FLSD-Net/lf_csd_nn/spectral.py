"""LFPR 與頻譜熵核心算子。"""

from __future__ import annotations

import jax.numpy as jnp


def compute_lfpr_and_entropy(
    v_window: jnp.ndarray,
    dct_matrix: jnp.ndarray,
    soft_low_mask: jnp.ndarray,
    *,
    normalize_entropy: bool = True,
    eps: float = 1e-15,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    spectrum = dct_matrix @ v_window
    power_spectrum = jnp.square(spectrum)
    low_energy = jnp.sum(power_spectrum * soft_low_mask)
    total_energy = jnp.sum(power_spectrum) + eps
    lfpr = low_energy / total_energy
    p_spectrum = power_spectrum / total_energy
    entropy = -jnp.sum(p_spectrum * jnp.log(p_spectrum + eps))
    if normalize_entropy:
        entropy = entropy / jnp.log(dct_matrix.shape[0])
    return lfpr, entropy
