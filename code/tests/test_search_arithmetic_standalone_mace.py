from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from search_arithmetic_standalone_mace import (  # noqa: E402
    candidate_for_prefix,
    canonical_pairs_sha256,
    compare_with_dense,
    load_mask_pairs,
    parse_mask_spec,
    ranked_layers,
)


def test_mask_loading_and_prefix_candidates_are_canonical(tmp_path: Path) -> None:
    path = tmp_path / "masks.npz"
    np.savez(
        path,
        mlp_seed=np.asarray([[1, 2], [0, 1], [1, 0], [0, 0]], dtype=np.int32),
    )
    lower = load_mask_pairs(path, key="mlp_seed", widths=[4, 4])
    upper = lower | {(0, 2), (1, 3)}
    first = candidate_for_prefix(lower, upper, [1, 0], 1)
    assert first == lower | {(1, 3)}
    assert candidate_for_prefix(lower, upper, [1, 0], 2) == upper
    assert canonical_pairs_sha256(first) == canonical_pairs_sha256(set(first))


def test_masks_must_cover_every_layer(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, mlp_final=np.asarray([[0, 0]], dtype=np.int32))
    with pytest.raises(ValueError, match="empty layers"):
        load_mask_pairs(path, key="mlp_final", widths=[2, 2])


def test_ranked_layers_filters_additions_and_appends_missing(tmp_path: Path) -> None:
    path = tmp_path / "ranking.json"
    path.write_text(
        json.dumps(
            {
                "rankings": {
                    "top_by_delta_per_1k": [
                        {"kind": "drop_layer", "layer": 1},
                        {"kind": "add_shell_layer", "layer": 2},
                        {"kind": "add_shell_layer", "layer": 0},
                        {"kind": "add_shell_layer", "layer": 2},
                    ]
                }
            }
        )
    )
    assert ranked_layers(path, num_layers=4) == [2, 0, 1, 3]


def test_dense_recovery_uses_only_dense_correct_denominator() -> None:
    dense = [
        {
            "id": "a",
            "prediction_text": "42",
            "exact_numeric_correct": True,
        },
        {
            "id": "b",
            "prediction_text": "wrong",
            "exact_numeric_correct": False,
        },
        {
            "id": "c",
            "prediction_text": "10",
            "exact_numeric_correct": True,
        },
    ]
    candidate = [
        {
            "id": "a",
            "prediction_text": "42",
            "exact_numeric_correct": True,
        },
        {
            "id": "b",
            "prediction_text": "11",
            "exact_numeric_correct": True,
        },
        {
            "id": "c",
            "prediction_text": "9",
            "exact_numeric_correct": False,
        },
    ]
    comparison = compare_with_dense(dense, candidate)
    assert comparison["dense_correct"] == 2
    assert comparison["matched_dense_correct"] == 1
    assert comparison["matched_dense_recovery"] == 0.5
    assert comparison["candidate_only_correct"] == 1
    assert comparison["dense_only_correct"] == 1


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("seed=/tmp/mask.npz", ("seed", Path("/tmp/mask.npz"), "mlp_final")),
        (
            "ceiling=/tmp/all.npz:mlp_rel_0.001",
            ("ceiling", Path("/tmp/all.npz"), "mlp_rel_0.001"),
        ),
    ],
)
def test_mask_spec_parser(
    spec: str,
    expected: tuple[str, Path, str],
) -> None:
    assert parse_mask_spec(spec) == expected
