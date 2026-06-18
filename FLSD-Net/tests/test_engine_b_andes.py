#!/usr/bin/env python3
"""
ANDES Kundur 驗證 — Engine B (DCT/JAX) vs FFT 參考
"""

from __future__ import annotations

import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from scipy.stats import spearmanr

from lf_csd_nn import (
    EngineBConfig,
    SpectralEngineB,
    build_engine_b_pipeline,
    compute_lfpr_and_entropy,
    dct_band_from_hz,
    generate_dct_matrix,
    generate_soft_masks,
)
from lf_csd_nn.andes_bridge import (
    F_LOW,
    FS,
    WINDOW_DEFAULT,
    reference_engine_b_score,
    reference_fft_lfpr_entropy,
    simulate_kundur,
)

warnings.filterwarnings("ignore", category=UserWarning)
jax.config.update("jax_enable_x64", True)

KUNDUR_IDX = list(range(0, 20))
KA_SWEEP = [10, 20, 50, 200]
WINDOW = WINDOW_DEFAULT


def test_dct_orthogonality() -> None:
    print("\n[TEST 1] DCT-II 正交性")
    n = 512
    dct = generate_dct_matrix(n)
    err = float(jnp.max(jnp.abs(dct @ dct.T - jnp.eye(n))))
    print(f"  max |MM^T - I| = {err:.2e}")
    assert err < 1e-10
    print("  [PASS]")


def test_lfpr_gradient() -> None:
    print("\n[TEST 2] LFPR 自動微分")
    n = WINDOW
    dct = generate_dct_matrix(n)
    lo, hi = dct_band_from_hz(n, FS, 0.1, F_LOW)
    low, _ = generate_soft_masks(n, lo, hi)

    def lfpr_fn(v):
        lf, _ = compute_lfpr_and_entropy(v, dct, low)
        return lf

    v = jax.random.normal(jax.random.PRNGKey(0), (n,))
    g = grad(lfpr_fn)(v)
    assert bool(jnp.isfinite(g).all())
    print("  grad finite = True")
    print("  [PASS]")


def test_synthetic_dct_vs_fft() -> None:
    print("\n[TEST 3] 合成信號 DCT vs FFT LFPR")
    n = WINDOW
    t = np.arange(n) / FS
    freqs_hz = [0.5, 1.0, 5.0, 15.0]
    engine = SpectralEngineB(EngineBConfig(window_size=n, fs=FS, f_low=F_LOW, normalize_entropy=False))
    engine._ensure_buffers()

    dct_lf, fft_lf = [], []
    for fh in freqs_hz:
        sig = np.sin(2 * np.pi * fh * t)
        lf_d, _ = compute_lfpr_and_entropy(
            jnp.asarray(sig), engine.dct_matrix, engine.low_mask, normalize_entropy=False
        )
        lf_f, _ = reference_fft_lfpr_entropy(sig)
        dct_lf.append(float(lf_d))
        fft_lf.append(lf_f)

    # 低頻正弦應有較高 LFPR；兩種變換趨勢一致即可（DCT bin ≠ FFT bin）
    assert dct_lf[0] > dct_lf[-1], f"DCT LFPR 非單調: {dct_lf}"
    assert fft_lf[0] > fft_lf[-1], f"FFT LFPR 非單調: {fft_lf}"
    rho, _ = spearmanr(dct_lf, fft_lf)
    print(f"  DCT LFPR = {[round(x, 4) for x in dct_lf]}")
    print(f"  FFT LFPR = {[round(x, 4) for x in fft_lf]}")
    print(f"  Spearman = {rho:+.3f}  (趨勢一致門檻 > 0.7)")
    assert rho > 0.7
    print("  [PASS]")


def test_andes_ka_sweep() -> None:
    print("\n[TEST 4] ANDES Kundur KA：DCT vs FFT")
    engine = SpectralEngineB(EngineBConfig(window_size=WINDOW, fs=FS, f_low=F_LOW, normalize_entropy=False))
    dct_scores, fft_scores = [], []
    for ka in KA_SWEEP:
        t, y = simulate_kundur(ka)
        s_dct, _ = engine.score_trajectory(t, y, KUNDUR_IDX)
        s_fft, _ = reference_engine_b_score(t, y, KUNDUR_IDX)
        dct_scores.append(s_dct)
        fft_scores.append(s_fft)
        print(f"  KA={ka:4d}  DCT={s_dct:.5f}  FFT={s_fft:.5f}")

    valid = ~np.isnan(dct_scores) & ~np.isnan(fft_scores)
    rho_pair, _ = spearmanr(np.array(dct_scores)[valid], np.array(fft_scores)[valid])
    print(f"  Spearman(DCT, FFT) = {rho_pair:+.3f}")
    assert rho_pair > 0.85
    print("  [PASS]")


def test_batch_smoke() -> None:
    print("\n[TEST 5] 1000 節點批次煙霧")
    pipeline = build_engine_b_pipeline(1000, window_size=512, fs=FS, f_low=F_LOW)
    data = jax.random.normal(jax.random.PRNGKey(42), (1000, 512))
    lfpr, ent = pipeline(data)
    assert lfpr.shape == (1000,) and bool(jnp.isfinite(lfpr).all())
    print(f"  LFPR[0]={float(lfpr[0]):.6f}")
    print("  [PASS]")


def run_all() -> None:
    print("=" * 60)
    print("LF CSD-NN — ANDES / FFT 驗證")
    print("=" * 60)
    test_dct_orthogonality()
    test_lfpr_gradient()
    test_synthetic_dct_vs_fft()
    test_andes_ka_sweep()
    test_batch_smoke()
    print("\n" + "=" * 60)
    print("全部測試通過")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_all()
    except Exception as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
