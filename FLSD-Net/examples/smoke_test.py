#!/usr/bin/env python3
"""煙霧測試：1000 節點批次 LFPR + 頻譜熵。"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lf_csd_nn import build_engine_b_pipeline

jax.config.update("jax_enable_x64", True)


def main() -> None:
    num_buses = 1000
    window_size = 512
    pipeline = build_engine_b_pipeline(num_buses, window_size=window_size)

    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, (num_buses, window_size))
    lfpr, entropy = pipeline(data)

    print("--- CSD Engine B (Spectral Radar) ---")
    print(f"Nodes     : {lfpr.shape[0]}")
    print(f"LFPR[0]   : {float(lfpr[0]):.6f}")
    print(f"Entropy[0]: {float(entropy[0]):.6f}")


if __name__ == "__main__":
    main()
