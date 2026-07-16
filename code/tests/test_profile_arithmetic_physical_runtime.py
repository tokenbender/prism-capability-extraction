from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from profile_arithmetic_physical_runtime import (  # noqa: E402
    _is_profiler_annotation,
    count_generated_tokens,
    estimate_execution_metrics,
    read_frozen_records,
    select_profile_records,
    summarize_cuda_kernels,
    summarize_operator_events,
    validate_outer_compile_settings,
    wrap_outer_compile,
)


def test_profiler_annotations_are_not_counted_as_cuda_kernel_launches() -> None:
    assert _is_profiler_annotation("arithmetic_runtime_generate")
    assert _is_profiler_annotation("Command Buffer Full")
    assert _is_profiler_annotation("## Call CompiledFxGraph abc123 ##")
    assert not _is_profiler_annotation("nvjet_sm100_tst_128x256")
    assert not _is_profiler_annotation("triton_poi_fused_silu_mul")


def test_outer_compile_settings_match_generation_compile_ownership_rule() -> None:
    validate_outer_compile_settings(
        outer_compile_mode="reduce-overhead",
        cache_implementation="dynamic",
        generation_disable_compile=False,
    )
    validate_outer_compile_settings(
        outer_compile_mode="reduce-overhead",
        cache_implementation="static",
        generation_disable_compile=True,
    )
    with pytest.raises(ValueError, match="both own generation"):
        validate_outer_compile_settings(
            outer_compile_mode="reduce-overhead",
            cache_implementation="static",
            generation_disable_compile=False,
        )
    with pytest.raises(ValueError, match="unsupported outer compile mode"):
        validate_outer_compile_settings(
            outer_compile_mode="invented",
            cache_implementation="dynamic",
            generation_disable_compile=False,
        )


def test_outer_compile_wrapper_records_wrap_time_and_requested_mode() -> None:
    model = object()
    calls = []

    def fake_compile(value: object, *, mode: str) -> tuple[object, str]:
        calls.append((value, mode))
        return value, mode

    compiled, seconds = wrap_outer_compile(
        model,
        mode="reduce-overhead",
        compile_factory=fake_compile,
    )
    untouched, no_compile_seconds = wrap_outer_compile(model, mode="none")

    assert compiled == (model, "reduce-overhead")
    assert calls == [(model, "reduce-overhead")]
    assert seconds >= 0.0
    assert untouched is model
    assert no_compile_seconds == 0.0


def test_frozen_records_are_read_exactly_and_selected_without_rerandomizing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    rows = [
        {
            "id": "compact",
            "prompt": "12 + 13 =",
            "answer": 25,
        },
        {
            "id": "spaced",
            "prompt": "52 + 53 =",
            "generation_prompt": "52 + 53 = ",
            "answer": 105,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = read_frozen_records(path)
    selected = select_profile_records(loaded, batch_size=1)

    assert loaded == rows
    assert selected == [rows[0]]
    assert loaded[1]["generation_prompt"] == "52 + 53 = "


def test_frozen_record_reader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            [
                {"id": "same", "prompt": "1 + 1 =", "answer": 2},
                {"id": "same", "prompt": "2 + 2 =", "answer": 4},
            ]
        )
    )

    with pytest.raises(ValueError, match="not unique"):
        read_frozen_records(path)


def test_operator_summary_captures_device_time_and_profiler_flop_estimates() -> None:
    events = [
        SimpleNamespace(
            key="aten::mm",
            count=2,
            self_device_time_total=30.0,
            device_time_total=40.0,
            flops=2_000_000_000_000.0,
            input_shapes=[[2, 3], [3, 4]],
        ),
        SimpleNamespace(
            key="aten::silu",
            count=1,
            self_device_time_total=50.0,
            device_time_total=55.0,
            flops=0.0,
            input_shapes=[[2, 4]],
        ),
        SimpleNamespace(
            key="cpu_only",
            count=1,
            self_device_time_total=0.0,
            device_time_total=0.0,
            flops=0.0,
            input_shapes=[],
        ),
    ]

    summary = summarize_operator_events(events, top_k=2)

    assert summary["profiler_estimated_executed_flops"] == 2_000_000_000_000.0
    assert summary["operator_rows_with_flop_estimate"] == 1
    assert summary["top_operators_by_self_device_time"][0]["name"] == (
        "aten::silu"
    )
    assert summary["top_operators_by_profiler_estimated_flops"][0][
        "name"
    ] == "aten::mm"


def test_kernel_summary_filters_transfers_and_estimates_summed_active_time() -> None:
    events = [
        SimpleNamespace(
            name="gemm_kernel",
            device_type="DeviceType.CUDA",
            self_device_time_total=2_000.0,
        ),
        SimpleNamespace(
            name="gemm_kernel",
            device_type="DeviceType.CUDA",
            self_device_time_total=1_000.0,
        ),
        SimpleNamespace(
            name="Memcpy DtoD",
            device_type="DeviceType.CUDA",
            self_device_time_total=9_000.0,
        ),
        SimpleNamespace(
            name="arithmetic_runtime_generate",
            device_type="DeviceType.CUDA",
            self_device_time_total=8_000.0,
        ),
        SimpleNamespace(
            name="cpu_op",
            device_type="DeviceType.CPU",
            self_device_time_total=7_000.0,
        ),
    ]

    summary = summarize_cuda_kernels(events, top_k=5)

    assert summary["cuda_device_event_count"] == 4
    assert summary["kernel_launch_count"] == 2
    assert summary[
        "profiler_estimated_summed_kernel_active_time_seconds"
    ] == pytest.approx(0.003)
    assert summary["top_cuda_kernels_by_device_time"] == [
        {
            "name": "gemm_kernel",
            "count": 2,
            "device_time_total_us": 3_000.0,
        }
    ]


def test_execution_metrics_label_wall_and_active_time_mfu_as_estimates() -> None:
    metrics = estimate_execution_metrics(
        profiler_estimated_flops=2_000_000_000_000.0,
        profiled_generation_seconds=2.0,
        hardware_peak_bf16_tflops=10.0,
        profiler_estimated_active_seconds=0.5,
    )

    assert metrics["estimate_only"] is True
    assert metrics["estimated_achieved_tflops_per_second"] == 1.0
    assert metrics["estimated_executed_mfu_fraction"] == pytest.approx(0.1)
    assert metrics["estimated_executed_mfu_percent"] == pytest.approx(10.0)
    assert metrics[
        "estimated_summed_kernel_active_to_wall_percent"
    ] == pytest.approx(25.0)
    assert metrics["estimated_active_time_tflops_per_second"] == 4.0
    assert metrics["estimated_active_time_mfu_percent"] == pytest.approx(40.0)
    assert "hardware-counter" in metrics["caveat"]
    assert "double-count" in metrics["active_time_caveat"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profiler_estimated_flops", -1.0, "FLOPs"),
        ("profiled_generation_seconds", 0.0, "seconds"),
        ("hardware_peak_bf16_tflops", 0.0, "hardware peak"),
    ],
)
def test_execution_metric_inputs_are_validated(
    field: str,
    value: float,
    message: str,
) -> None:
    kwargs = {
        "profiler_estimated_flops": 1.0,
        "profiled_generation_seconds": 1.0,
        "hardware_peak_bf16_tflops": 1.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        estimate_execution_metrics(**kwargs)


def test_generated_token_receipt_separates_slots_from_non_stop_tokens() -> None:
    sequences = torch.tensor(
        [
            [10, 11, 25, 99, 0],
            [10, 11, 105, 99, 0],
        ]
    )

    result = count_generated_tokens(
        sequences,
        prompt_width=2,
        stop_ids={99},
        pad_token_id=0,
    )

    assert result == {
        "generated_token_slots": 6,
        "non_stop_generated_tokens": 2,
    }
