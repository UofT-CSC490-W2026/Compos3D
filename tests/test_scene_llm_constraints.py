from __future__ import annotations

from compos3d.llm.scene_llm import _normalize_constraints


def test_normalize_constraints_ignores_blank_items() -> None:
    assert _normalize_constraints(["keep rug centered", "   ", {"text": ""}]) == [
        {"text": "keep rug centered"}
    ]


def test_normalize_constraints_allows_empty_after_filtering() -> None:
    assert _normalize_constraints(["   ", {"text": " "}]) == []
