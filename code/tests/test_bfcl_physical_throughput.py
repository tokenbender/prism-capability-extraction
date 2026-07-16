from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_bfcl_physical_throughput import (  # noqa: E402
    accepted_generation_tokens,
    percentile,
    summarize,
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
