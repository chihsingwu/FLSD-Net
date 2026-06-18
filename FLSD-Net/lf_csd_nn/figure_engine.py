"""
LF CSD-NN 圖表引擎 — 產生 README / 論文用驗證圖。

Fig 1  KA 掃描：DCT vs FFT LFPR
Fig 2  故障後 LFPR 時序（KA=10 vs 200）
Fig 3  合成單頻：DCT vs FFT LFPR 對照
Fig 4  DCT 功率譜 + 柔性低頻遮罩示意
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from . import EngineBConfig, SpectralEngineB, compute_lfpr_and_entropy
from .andes_bridge import (
    F_LOW,
    FS,
    WINDOW_DEFAULT,
    reference_engine_b_score,
    reference_fft_lfpr_entropy,
    simulate_kundur,
)

jax.config.update("jax_enable_x64", True)

KUNDUR_IDX = list(range(0, 20))

# ANDES 不可用時的備援（與 test_engine_b_andes 實測一致）
FALLBACK_KA = {
    10: {"dct": 0.26653, "fft": 0.13625},
    20: {"dct": 0.33175, "fft": 0.22253},
    50: {"dct": 0.42120, "fft": 0.35572},
    100: {"dct": 0.44500, "fft": 0.39500},
    200: {"dct": 0.46894, "fft": 0.43430},
}


@dataclass
class FigureEngineConfig:
    output_dir: Path = field(default_factory=lambda: Path("docs/figures"))
    ka_sweep: tuple[int, ...] = (10, 20, 50, 100, 200)
    ka_compare: tuple[int, int] = (10, 200)
    kundur_tf: float = 12.0
    dpi: int = 160
    engine_b: EngineBConfig = field(
        default_factory=lambda: EngineBConfig(
            window_size=WINDOW_DEFAULT, fs=FS, f_low=F_LOW, normalize_entropy=False
        )
    )


class FigureEngine:
    """CSD Engine B 驗證圖表批次產生器。"""

    def __init__(self, config: FigureEngineConfig | None = None):
        self.config = config or FigureEngineConfig()
        self._engine = SpectralEngineB(self.config.engine_b)
        self._andes_ok: bool | None = None

    def _out(self) -> Path:
        p = Path(self.config.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _has_andes(self) -> bool:
        if self._andes_ok is not None:
            return self._andes_ok
        try:
            import andes  # noqa: F401

            self._andes_ok = True
        except ImportError:
            self._andes_ok = False
        return self._andes_ok

    def collect_ka_scores(self) -> tuple[list[int], list[float], list[float]]:
        kas = list(self.config.ka_sweep)
        dct_s, fft_s = [], []
        if not self._has_andes():
            for ka in kas:
                fb = FALLBACK_KA.get(ka, {"dct": np.nan, "fft": np.nan})
                dct_s.append(fb["dct"])
                fft_s.append(fb["fft"])
            return kas, dct_s, fft_s

        for ka in kas:
            t, y = simulate_kundur(ka, tf=self.config.kundur_tf)
            d, _ = self._engine.score_trajectory(t, y, KUNDUR_IDX)
            f, _ = reference_engine_b_score(t, y, KUNDUR_IDX)
            dct_s.append(d)
            fft_s.append(f)
        return kas, dct_s, fft_s

    def _lfpr_timeseries(
        self, ka: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """回傳 (t_mid, lfpr_dct, lfpr_fft) 滑窗序列。"""
        if not self._has_andes():
            return None
        cfg = self.config.engine_b
        t, y = simulate_kundur(ka, tf=self.config.kundur_tf)
        y_sub = y[:, KUNDUR_IDX].astype(np.float64)
        y_sub = (y_sub - y_sub.mean(0)) / (y_sub.std(0) + 1e-10)
        self._engine._ensure_buffers()
        window, stride = cfg.window_size, cfg.stride

        t_mid, dct_v, fft_v = [], [], []
        for start in range(0, len(t) - window, stride):
            mid = t[start + window // 2]
            if mid < cfg.fault_t:
                continue
            seg = y_sub[start : start + window]
            lf_d, _ = self._engine.lfpr_entropy_batch(seg.T)
            ch_fft = [reference_fft_lfpr_entropy(seg[:, c])[0] for c in range(seg.shape[1])]
            t_mid.append(mid)
            dct_v.append(float(np.mean(lf_d)))
            fft_v.append(float(np.mean(ch_fft)))
        return np.array(t_mid), np.array(dct_v), np.array(fft_v)

    def fig1_ka_lfpr_dct_vs_fft(self) -> Path:
        kas, dct_s, fft_s = self.collect_ka_scores()
        rho, _ = spearmanr(dct_s, fft_s)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(kas, dct_s, "o-", color="#2563eb", lw=2, ms=8, label="Engine B DCT (JAX)")
        ax.plot(kas, fft_s, "s--", color="#dc2626", lw=2, ms=7, label="Reference FFT")
        ax.set_xlabel("Exciter gain KA")
        ax.set_ylabel("LFPR score (post-fault mean)")
        ax.set_title(f"Kundur: DCT vs FFT LFPR (Spearman={rho:.2f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self._out() / "fig1_ka_lfpr_dct_vs_fft.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def fig2_lfpr_timeseries(self) -> Path | None:
        ka_lo, ka_hi = self.config.ka_compare
        series = {}
        for ka, color, label in [
            (ka_lo, "#16a34a", f"KA={ka_lo} (stronger damping)"),
            (ka_hi, "#dc2626", f"KA={ka_hi} (weaker damping)"),
        ]:
            ts = self._lfpr_timeseries(ka)
            if ts is None:
                return None
            series[ka] = (ts, color, label)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, kind, ylab in zip(axes, ("dct", "fft"), ("DCT LFPR", "FFT LFPR")):
            for ka in (ka_lo, ka_hi):
                (t_mid, dct_v, fft_v), color, label = series[ka]
                y = dct_v if kind == "dct" else fft_v
                ax.plot(t_mid, y, color=color, lw=1.5, label=label)
            ax.axvline(1.0, color="#64748b", ls="--", lw=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylab)
            ax.set_title(ylab)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Kundur post-fault LFPR sliding windows", fontsize=11)
        fig.tight_layout()
        path = self._out() / "fig2_lfpr_timeseries.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def fig3_synthetic_dct_vs_fft(self) -> Path:
        n = WINDOW_DEFAULT
        t = np.arange(n) / FS
        freqs = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
        self._engine._ensure_buffers()
        dct_lf, fft_lf = [], []
        for fh in freqs:
            sig = np.sin(2 * np.pi * fh * t)
            ld, _ = compute_lfpr_and_entropy(
                jnp.asarray(sig),
                self._engine.dct_matrix,
                self._engine.low_mask,
                normalize_entropy=False,
            )
            lf, _ = reference_fft_lfpr_entropy(sig)
            dct_lf.append(float(ld))
            fft_lf.append(float(lf))

        x = np.arange(len(freqs))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(x - w / 2, dct_lf, w, label="DCT (JAX)", color="#2563eb", alpha=0.85)
        ax.bar(x + w / 2, fft_lf, w, label="FFT (ref)", color="#dc2626", alpha=0.85)
        ax.set_xticks(x, [f"{f} Hz" for f in freqs])
        ax.set_ylabel("LFPR")
        ax.set_title(f"Synthetic sinusoids (f_low={F_LOW} Hz, fs={FS} Hz)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = self._out() / "fig3_synthetic_dct_vs_fft.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def fig4_dct_mask_spectrum(self) -> Path:
        """白噪聲 DCT 功率譜 + 柔性遮罩（教學用）。"""
        from .dct import dct_band_from_hz, generate_dct_matrix
        from .masks import generate_soft_masks

        n = 512
        low_b, high_b = dct_band_from_hz(n, FS, 0.1, F_LOW, exclude_dc=True)
        dct = generate_dct_matrix(n)
        mask, _ = generate_soft_masks(n, low_b, high_b, smoothness=1.5)
        key = jax.random.PRNGKey(0)
        v = jax.random.normal(key, (n,))
        spec_pow = np.asarray(jnp.square(dct @ v))
        mask = np.asarray(mask)
        bins = np.arange(n)
        freqs_hz = bins * FS / (2 * n)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(freqs_hz, 0, spec_pow / spec_pow.max(), alpha=0.25, color="#94a3b8", label="|DCT|^2 (norm)")
        ax.plot(freqs_hz, mask, color="#2563eb", lw=2, label="Soft low-band mask")
        ax.axvline(F_LOW, color="#dc2626", ls="--", lw=1.2, label=f"f_low={F_LOW} Hz")
        ax.set_xlim(0, min(30, freqs_hz[-1]))
        ax.set_xlabel("Approx. frequency (Hz)")
        ax.set_ylabel("Normalized power / mask")
        ax.set_title("DCT power spectrum & differentiable LF band mask")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self._out() / "fig4_dct_mask_spectrum.png"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_all(self) -> list[Path]:
        paths: list[Path] = []
        paths.append(self.fig1_ka_lfpr_dct_vs_fft())
        p2 = self.fig2_lfpr_timeseries()
        if p2 is not None:
            paths.append(p2)
        paths.append(self.fig3_synthetic_dct_vs_fft())
        paths.append(self.fig4_dct_mask_spectrum())
        return paths
