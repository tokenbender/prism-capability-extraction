from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_bfcl_physical_throughput import (  # noqa: E402
    CompiledDecodeCudaTimer,
    accepted_generation_tokens,
    counter_delta,
    percentile,
    summarize,
    validate_compile_layering,
)
from load_bfcl_physical_bundle import (  # noqa: E402
    build_generation_compile_settings,
)


def test_accepted_generation_tokens_stops_at_first_eos() -> None:
    sequences = torch.tensor(
        [
            [11, 12, 21, 22, 99, 99],
            [11, 12, 31, 99, 99, 99],
            [11, 12, 41, 42, 43, 44],
        ]
    )

    assert accepted_generation_tokens(
        sequences,
        prompt_width=2,
        eos_token_ids={99},
    ) == 3 + 2 + 4


def test_summary_records_tail_and_dispersion() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0]

    assert percentile(values, 0.95) == pytest.approx(8.8)
    result = summarize(values)
    assert result["median"] == 3.0
    assert result["p95"] == pytest.approx(8.8)
    assert result["max"] == 10.0
    assert result["stdev"] > 0


def test_summary_supports_batch_latency_receipts() -> None:
    latencies_ms = [10.0, 12.0, 11.0, 17.0]

    result = summarize(latencies_ms)

    assert result["median"] == pytest.approx(11.5)
    assert result["p95"] == pytest.approx(16.25)
    assert result["min"] == 10.0


def test_compiled_decode_timer_summarizes_device_slots() -> None:
    class FakeEvent:
        def __init__(self, timestamp_ms: float) -> None:
            self.timestamp_ms = timestamp_ms

        def elapsed_time(self, other: "FakeEvent") -> float:
            return other.timestamp_ms - self.timestamp_ms

    timer = CompiledDecodeCudaTimer(lambda: None)
    timer.records = [  # type: ignore[assignment]
        (FakeEvent(1.0), FakeEvent(3.0), 64),
        (FakeEvent(4.0), FakeEvent(7.0), 64),
    ]

    assert timer.summarize_since(0) == {
        "calls": 2,
        "input_slots": 128,
        "device_seconds": pytest.approx(0.005),
        "input_slots_per_second": pytest.approx(25_600.0),
    }
    assert timer.summarize_since(1)["input_slots"] == 64
    with pytest.raises(ValueError, match="out of range"):
        timer.summarize_since(3)


def test_counter_delta_records_measurement_only_changes() -> None:
    assert counter_delta(None, {"unique_graphs": 2}) is None
    assert counter_delta(
        {"calls_captured": 100, "unique_graphs": 2},
        {"calls_captured": 100, "unique_graphs": 2, "new": 0},
    ) == {"calls_captured": 0, "new": 0, "unique_graphs": 0}


def test_generation_compile_can_be_explicitly_disabled_for_static_cache() -> None:
    kwargs, receipt = build_generation_compile_settings(
        cache_implementation="static",
        disable_compile=True,
        compile_dynamic=False,
    )

    assert kwargs == {"disable_compile": True}
    assert receipt == {
        "mode": "disabled",
        "disable_compile": True,
        "compile_config": None,
    }


def test_generation_dynamic_compile_is_explicit_and_receipted() -> None:
    captured: dict[str, object] = {}

    def fake_compile_config(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    kwargs, receipt = build_generation_compile_settings(
        cache_implementation="static",
        disable_compile=False,
        compile_dynamic=True,
        compile_config_factory=fake_compile_config,
    )

    assert set(kwargs) == {"compile_config"}
    assert captured == {
        "mode": "reduce-overhead",
        "dynamic": True,
        "fullgraph": False,
    }
    assert receipt == {
        "mode": "dynamic_reduce_overhead",
        "disable_compile": False,
        "compile_config": {
            **captured,
            "backend": "inductor",
            "options": None,
        },
    }


def test_static_auto_compile_receipts_resolved_transformers_defaults() -> None:
    kwargs, receipt = build_generation_compile_settings(
        cache_implementation="static",
        disable_compile=False,
        compile_dynamic=False,
    )

    assert kwargs == {}
    assert receipt == {
        "mode": "transformers_default_auto",
        "disable_compile": None,
        "compile_config": {
            "source": "transformers_default",
            "mode": "reduce-overhead",
            "dynamic": None,
            "fullgraph": False,
            "backend": "inductor",
            "options": None,
        },
    }


@pytest.mark.parametrize(
    ("disable_compile", "compile_dynamic", "message"),
    [
        (True, True, "mutually exclusive"),
        (True, False, "require --cache-implementation static"),
        (False, True, "require --cache-implementation static"),
    ],
)
def test_generation_compile_controls_reject_ambiguous_dynamic_cache_modes(
    disable_compile: bool,
    compile_dynamic: bool,
    message: str,
) -> None:
    cache_implementation = "static" if disable_compile and compile_dynamic else "dynamic"
    with pytest.raises(ValueError, match=message):
        build_generation_compile_settings(
            cache_implementation=cache_implementation,
            disable_compile=disable_compile,
            compile_dynamic=compile_dynamic,
        )


def test_outer_and_generation_compile_cannot_both_own_static_cache() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_compile_layering(
            outer_compile_mode="reduce-overhead",
            cache_implementation="static",
            generation_disable_compile=False,
        )

    validate_compile_layering(
        outer_compile_mode="reduce-overhead",
        cache_implementation="static",
        generation_disable_compile=True,
    )
