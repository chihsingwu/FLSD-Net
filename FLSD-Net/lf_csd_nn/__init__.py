"""
LF CSD-NN — 電力系統 CSD Engine B（可微 LFPR + 頻譜熵）
"""

from .dct import dct_band_from_hz, generate_dct_matrix, hz_to_dct_index
from .engine import EngineBConfig, SpectralEngineB, build_engine_b_pipeline
from .figure_engine import FigureEngine, FigureEngineConfig
from .masks import generate_soft_masks
from .park import abc_to_dq
from .spectral import compute_lfpr_and_entropy

__all__ = [
    "abc_to_dq",
    "generate_dct_matrix",
    "hz_to_dct_index",
    "dct_band_from_hz",
    "generate_soft_masks",
    "compute_lfpr_and_entropy",
    "EngineBConfig",
    "SpectralEngineB",
    "build_engine_b_pipeline",
    "FigureEngine",
    "FigureEngineConfig",
]

__version__ = "0.1.0"
