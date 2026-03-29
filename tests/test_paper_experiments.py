from __future__ import annotations

from compos3d.paper.experiments import _summarize_generation_pairwise


def test_summarize_generation_pairwise_computes_rates_and_scores() -> None:
    rows = [
        {
            "method_a": "hyp",
            "method_b": "baseline",
            "asset_selection": "A",
            "layout_coherence": "B",
            "overall_preference": "tie",
        },
        {
            "method_a": "hyp",
            "method_b": "baseline",
            "asset_selection": "A",
            "layout_coherence": "A",
            "overall_preference": "A",
        },
        {
            "method_a": "hyp",
            "method_b": "baseline",
            "asset_selection": "B",
            "layout_coherence": "tie",
            "overall_preference": "B",
        },
    ]

    summary = _summarize_generation_pairwise(rows)

    assert summary["num_examples"] == 3
    assert summary["method_a"] == "hyp"
    assert summary["method_b"] == "baseline"
    assert summary["headline_metrics"] == {
        "generation_quality_score": 0.5,
        "generation_quality_win_rate": 0.3333,
        "generation_quality_non_tie_win_rate": 0.5,
        "asset_selection_score": 0.6667,
        "layout_coherence_score": 0.5,
    }

    assert summary["asset_selection"]["counts"] == {"A": 2, "B": 1}
    assert summary["asset_selection"]["preferred_win_rate"] == 0.6667
    assert summary["asset_selection"]["other_win_rate"] == 0.3333
    assert summary["asset_selection"]["tie_rate"] == 0.0
    assert summary["asset_selection"]["preferred_score"] == 0.6667
    assert summary["asset_selection"]["preferred_non_tie_win_rate"] == 0.6667

    assert summary["layout_coherence"]["counts"] == {"B": 1, "A": 1, "tie": 1}
    assert summary["layout_coherence"]["preferred_score"] == 0.5
    assert summary["layout_coherence"]["preferred_non_tie_win_rate"] == 0.5

    assert summary["overall_preference"]["counts"] == {"tie": 1, "A": 1, "B": 1}
    assert summary["overall_preference"]["preferred_score"] == 0.5
    assert summary["overall_preference"]["tie_rate"] == 0.3333
