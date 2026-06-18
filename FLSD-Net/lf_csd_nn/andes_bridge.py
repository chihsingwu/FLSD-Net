"""ANDES 仿真與 FFT 參考（供測試與繪圖共用）。"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, fftfreq

FS = 60.0
F_LOW = 2.0
WINDOW_DEFAULT = 120


def andes_case_path() -> str:
    import andes

    return andes.get_case("kundur/kundur_full.xlsx")


def simulate_kundur(
    ka: float, tf: float = 15.0, bus: int = 7, tc: float = 1.1, fs: float = FS
):
    import andes

    ss = andes.load(andes_case_path(), setup=False, no_output=True)
    n = len(ss.EXDC2.KA.v)
    ss.EXDC2.KA.v = [float(ka)] * n
    ss.add("Fault", {"idx": "F1", "bus": bus, "tf": 1.0, "tc": tc, "xf": 0.01})
    ss.setup()
    ss.PFlow.run()
    ss.TDS.config.tf = tf
    ss.TDS.config.tstep = 1 / fs
    ss.TDS.run()
    return np.asarray(ss.dae.ts.t), np.asarray(ss.dae.ts.y)


def reference_fft_lfpr_entropy(seg: np.ndarray, fs: float = FS, f_low: float = F_LOW):
    window = len(seg)
    freqs = fftfreq(window, d=1.0 / fs)
    low_mask = (np.abs(freqs) > 0) & (np.abs(freqs) <= f_low)
    psd = np.abs(fft(seg)[: window // 2]) ** 2
    total = psd.sum() + 1e-12
    lfpr = float(psd[low_mask[: window // 2]].sum() / total)
    p = psd / total
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    return lfpr, ent


def reference_engine_b_score(
    t, y, idx, *, fault_t=1.0, window=WINDOW_DEFAULT, stride=10, fs=FS, f_low=F_LOW
):
    if len(t) < window + 10:
        return float("nan"), float("nan")
    y_sub = (y[:, idx].astype(np.float64) - y[:, idx].mean(0)) / (y[:, idx].std(0) + 1e-10)
    lfpr_v, ent_v = [], []
    for start in range(0, len(t) - window, stride):
        if t[start + window // 2] < fault_t:
            continue
        seg = y_sub[start : start + window]
        ch_lf, ch_en = [], []
        for c in range(seg.shape[1]):
            lf, en = reference_fft_lfpr_entropy(seg[:, c], fs=fs, f_low=f_low)
            ch_lf.append(lf)
            ch_en.append(en)
        lfpr_v.append(np.mean(ch_lf))
        ent_v.append(np.mean(ch_en))
    if len(lfpr_v) < 5:
        return float("nan"), float("nan")
    return float(np.mean(lfpr_v[-20:])), float(np.mean(ent_v[-20:]))
