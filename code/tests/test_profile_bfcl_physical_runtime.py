from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from profile_bfcl_physical_runtime import _is_profiler_annotation  # noqa: E402


def test_profiler_annotations_are_not_counted_as_cuda_kernel_launches() -> None:
    assert _is_profiler_annotation("physical_bfcl_generate")
    assert _is_profiler_annotation("Command Buffer Full")
    assert _is_profiler_annotation("## Call CompiledFxGraph abc123 ##")
    assert not _is_profiler_annotation("nvjet_sm100_tst_128x256")
    assert not _is_profiler_annotation("triton_poi_fused_silu_mul")
