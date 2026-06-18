#!/usr/bin/env python3
"""
LF CSD-NN 圖表引擎 CLI
======================
  Fig 1  KA 掃描 DCT vs FFT
  Fig 2  故障後 LFPR 時序（需 andes）
  Fig 3  合成單頻 LFPR 對照
  Fig 4  DCT 功率譜 + 柔性遮罩

執行：
  pip install andes matplotlib
  python examples/generate_figures.py
  python examples/generate_figures.py --out docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lf_csd_nn.figure_engine import FigureEngine, FigureEngineConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="LF CSD-NN figure engine")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "figures",
        help="輸出目錄",
    )
    args = ap.parse_args()

    engine = FigureEngine(FigureEngineConfig(output_dir=args.out))
    paths = engine.generate_all()
    print("--- LF CSD-NN Figure Engine ---")
    for p in paths:
        print(f"  Saved: {p}")
    print(f"Total: {len(paths)} figures")


if __name__ == "__main__":
    main()
