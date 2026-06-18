"""
向後相容入口。請改用 `from lf_csd_nn import ...`。
"""

from lf_csd_nn import (
    EngineBConfig,
    SpectralEngineB,
    abc_to_dq,
    build_engine_b_pipeline,
    compute_lfpr_and_entropy,
    generate_dct_matrix,
)

__all__ = [
    "abc_to_dq",
    "generate_dct_matrix",
    "compute_lfpr_and_entropy",
    "EngineBConfig",
    "SpectralEngineB",
    "build_engine_b_pipeline",
]

if __name__ == "__main__":
    from examples.smoke_test import main

    main()
