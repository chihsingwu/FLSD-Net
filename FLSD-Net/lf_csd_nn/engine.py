"""Engine B：批次 LFPR / 頻譜熵管線。"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from .dct import dct_band_from_hz, generate_dct_matrix
from .masks import generate_soft_masks
from .spectral import compute_lfpr_and_entropy


@dataclass
class EngineBConfig:
    window_size: int = 120
    fs: float = 60.0
    f_low: float = 2.0
    f_high: float | None = None
    mask_smoothness: float = 2.0
    normalize_entropy: bool = True
    stride: int = 10
    fault_t: float = 1.0
    score_tail: int = 20


class SpectralEngineB:
    def __init__(self, config: EngineBConfig | None = None):
        self.config = config or EngineBConfig()
        self._dct_matrix = None
        self._low_mask = None
        self._pipeline = None

    def _ensure_buffers(self) -> None:
        if self._dct_matrix is not None:
            return
        cfg = self.config
        n = cfg.window_size
        f_high = cfg.f_high if cfg.f_high is not None else cfg.f_low
        low_bound, high_bound = dct_band_from_hz(
            n, cfg.fs, max(cfg.f_low * 0.05, 0.1), f_high, exclude_dc=True
        )
        self._dct_matrix = generate_dct_matrix(n)
        self._low_mask, _ = generate_soft_masks(
            n, low_bound, high_bound, smoothness=cfg.mask_smoothness
        )
        dct, low, norm_ent = self._dct_matrix, self._low_mask, cfg.normalize_entropy

        @jit
        def _batch(batch_v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return vmap(
                lambda v: compute_lfpr_and_entropy(
                    v, dct, low, normalize_entropy=norm_ent
                )
            )(batch_v)

        self._pipeline = _batch

    @property
    def dct_matrix(self) -> jnp.ndarray:
        self._ensure_buffers()
        return self._dct_matrix

    @property
    def low_mask(self) -> jnp.ndarray:
        self._ensure_buffers()
        return self._low_mask

    def lfpr_entropy_batch(self, batch_v: np.ndarray | jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_buffers()
        lfpr, ent = self._pipeline(jnp.asarray(batch_v, dtype=jnp.float64))
        return np.asarray(lfpr), np.asarray(ent)

    def score_trajectory(
        self, t: np.ndarray, y: np.ndarray, channel_idx: list[int], *, fault_t: float | None = None
    ) -> tuple[float, float]:
        cfg = self.config
        fault_t = cfg.fault_t if fault_t is None else fault_t
        window, stride = cfg.window_size, cfg.stride
        if len(t) < window + 10:
            return float("nan"), float("nan")
        y_sub = y[:, channel_idx].astype(np.float64)
        y_sub = (y_sub - y_sub.mean(0)) / (y_sub.std(0) + 1e-10)
        self._ensure_buffers()
        lfpr_v, ent_v = [], []
        for start in range(0, len(t) - window, stride):
            if t[start + window // 2] < fault_t:
                continue
            seg = y_sub[start : start + window].T
            lf, en = self.lfpr_entropy_batch(seg)
            lfpr_v.append(float(np.mean(lf)))
            ent_v.append(float(np.mean(en)))
        if len(lfpr_v) < 5:
            return float("nan"), float("nan")
        return float(np.mean(lfpr_v[-cfg.score_tail :])), float(np.mean(ent_v[-cfg.score_tail :]))


def build_engine_b_pipeline(
    num_buses: int,
    window_size: int = 512,
    fs: float = 60.0,
    f_low: float = 2.0,
    mask_smoothness: float = 1.5,
):
    del num_buses
    low_b, high_b = dct_band_from_hz(window_size, fs, 0.1, f_low, exclude_dc=True)
    dct = generate_dct_matrix(window_size)
    low_mask, _ = generate_soft_masks(window_size, low_b, high_b, mask_smoothness)

    @jit
    def pipeline(batch_v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return vmap(
            lambda v: compute_lfpr_and_entropy(v, dct, low_mask, normalize_entropy=True)
        )(batch_v)

    return pipeline
