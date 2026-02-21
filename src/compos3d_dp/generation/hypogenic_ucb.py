"""
HypoGeniC-style UCB Bandit for Compositional Rule Discovery

Implements Upper Confidence Bound (UCB) bandit algorithm to:
1. Discover compositional rules for 3D scenes
2. Select which rules to apply based on success rates
3. Evolve hypothesis bank over time

Based on: external/hypothesis-generation/hypothesis_agent/

Key concepts:
- **Hypothesis**: A compositional rule (e.g., "Add red materials to increase realism")
- **UCB**: Balance exploration (try new hypotheses) vs exploitation (use best hypotheses)
- **Bandit**: Multi-armed bandit where each arm is a hypothesis
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math


@dataclass
class Hypothesis:
    """A compositional rule for 3D scene generation"""

    hypothesis_id: str
    text: str  # Natural language description
    category: str  # e.g., "lighting", "camera", "materials", "composition"

    # UCB statistics
    num_trials: int = 0
    num_successes: int = 0
    total_reward: float = 0.0

    # Metadata
    created_at: str = ""
    last_used: str = ""
    confidence: float = 0.0

    @property
    def success_rate(self) -> float:
        """Empirical success rate"""
        if self.num_trials == 0:
            return 0.0
        return self.num_successes / self.num_trials

    @property
    def avg_reward(self) -> float:
        """Average reward"""
        if self.num_trials == 0:
            return 0.0
        return self.total_reward / self.num_trials

    def ucb_score(self, total_trials: int, exploration_weight: float = 2.0) -> float:
        """
        Compute UCB1 score.

        UCB1 = avg_reward + exploration_weight * sqrt(log(total_trials) / num_trials)

        Args:
            total_trials: Total trials across all hypotheses
            exploration_weight: Controls exploration vs exploitation

        Returns:
            UCB score (higher = should select this hypothesis)
        """
        if self.num_trials == 0:
            return float("inf")  # Always try untested hypotheses first

        exploitation = self.avg_reward
        exploration = exploration_weight * math.sqrt(
            math.log(total_trials + 1) / self.num_trials
        )

        return exploitation + exploration

    def update(self, reward: float, success: bool):
        """Update statistics after using this hypothesis"""
        self.num_trials += 1
        self.total_reward += reward
        if success:
            self.num_successes += 1
        self.last_used = datetime.now(timezone.utc).isoformat()


class HypothesisBank:
    """
    Manages a bank of hypotheses with UCB selection.

    Usage:
        bank = HypothesisBank()
        bank.add_hypothesis("Use warm lighting for cozy scenes", "lighting")

        # Select best hypothesis
        hyp = bank.select_hypothesis()

        # After using hypothesis
        bank.update_hypothesis(hyp.hypothesis_id, reward=0.85, success=True)

        # Save/load
        bank.save("hypothesis_bank.json")
    """

    def __init__(self, exploration_weight: float = 2.0):
        """
        Initialize hypothesis bank.

        Args:
            exploration_weight: UCB exploration parameter (higher = more exploration)
        """
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.exploration_weight = exploration_weight
        self.total_trials = 0

        # Initialize with some default hypotheses
        self._initialize_default_hypotheses()

    def _initialize_default_hypotheses(self):
        """Add default compositional rules"""
        defaults = [
            # Lighting
            ("Use three-point lighting for professional look", "lighting"),
            ("Add warm lighting for cozy atmosphere", "lighting"),
            ("Use dramatic shadows for mood", "lighting"),
            ("Add ambient occlusion for depth", "lighting"),
            # Camera
            ("Use rule of thirds for composition", "camera"),
            ("Lower camera angle for dramatic effect", "camera"),
            ("Wide angle for spacious feel", "camera"),
            ("Close-up for detail focus", "camera"),
            # Materials
            ("Use PBR materials for realism", "materials"),
            ("Add roughness variation for natural look", "materials"),
            ("Use complementary colors", "materials"),
            ("Add metallic accents", "materials"),
            # Composition
            ("Balance large and small objects", "composition"),
            ("Use negative space effectively", "composition"),
            ("Create visual hierarchy", "composition"),
            ("Add foreground elements for depth", "composition"),
            # Physics
            ("Ensure objects don't overlap", "physics"),
            ("Use realistic scale relationships", "physics"),
            ("Add supporting surfaces", "physics"),
            ("Respect gravity and balance", "physics"),
        ]

        for text, category in defaults:
            self.add_hypothesis(text, category)

    def add_hypothesis(self, text: str, category: str) -> Hypothesis:
        """Add a new hypothesis to the bank"""
        hypothesis_id = f"{category}_{len([h for h in self.hypotheses.values() if h.category == category])}"

        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            text=text,
            category=category,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.hypotheses[hypothesis_id] = hypothesis
        return hypothesis

    def select_hypothesis(
        self,
        category: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> Hypothesis:
        """
        Select hypothesis using UCB1 algorithm.

        Args:
            category: Only select from this category (None = all)
            exclude_ids: Don't select these hypotheses

        Returns:
            Selected hypothesis
        """
        candidates = [
            h
            for h in self.hypotheses.values()
            if (category is None or h.category == category)
            and (exclude_ids is None or h.hypothesis_id not in exclude_ids)
        ]

        if not candidates:
            raise ValueError(f"No hypotheses available (category={category})")

        # Compute UCB scores
        best_hyp = max(
            candidates,
            key=lambda h: h.ucb_score(self.total_trials, self.exploration_weight),
        )

        return best_hyp

    def select_top_k(
        self, k: int = 3, category: Optional[str] = None
    ) -> List[Hypothesis]:
        """Select top k hypotheses by UCB score"""
        candidates = [
            h
            for h in self.hypotheses.values()
            if (category is None or h.category == category)
        ]

        sorted_hyps = sorted(
            candidates,
            key=lambda h: h.ucb_score(self.total_trials, self.exploration_weight),
            reverse=True,
        )

        return sorted_hyps[:k]

    def update_hypothesis(self, hypothesis_id: str, reward: float, success: bool):
        """Update hypothesis after using it"""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis not found: {hypothesis_id}")

        self.hypotheses[hypothesis_id].update(reward, success)
        self.total_trials += 1

    def get_best_hypotheses(self, top_k: int = 5) -> List[Hypothesis]:
        """Get best performing hypotheses by average reward"""
        sorted_hyps = sorted(
            self.hypotheses.values(), key=lambda h: h.avg_reward, reverse=True
        )
        return sorted_hyps[:top_k]

    def get_statistics(self) -> Dict:
        """Get summary statistics"""
        by_category = {}
        for h in self.hypotheses.values():
            if h.category not in by_category:
                by_category[h.category] = []
            by_category[h.category].append(h)

        stats = {
            "total_hypotheses": len(self.hypotheses),
            "total_trials": self.total_trials,
            "by_category": {},
        }

        for cat, hyps in by_category.items():
            stats["by_category"][cat] = {
                "count": len(hyps),
                "avg_success_rate": np.mean(
                    [h.success_rate for h in hyps if h.num_trials > 0]
                ),
                "avg_trials": np.mean([h.num_trials for h in hyps]),
            }

        return stats

    def save(self, path: str | Path):
        """Save hypothesis bank to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "exploration_weight": self.exploration_weight,
            "total_trials": self.total_trials,
            "hypotheses": [asdict(h) for h in self.hypotheses.values()],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HypothesisBank":
        """Load hypothesis bank from JSON"""
        with open(path) as f:
            data = json.load(f)

        bank = cls(exploration_weight=data["exploration_weight"])
        bank.total_trials = data["total_trials"]
        bank.hypotheses = {}

        for h_data in data["hypotheses"]:
            h = Hypothesis(**h_data)
            bank.hypotheses[h.hypothesis_id] = h

        print(f"📂 Loaded hypothesis bank from {path}")
        return bank


if __name__ == "__main__":
    # Test the UCB bandit
    print("🧪 Testing HypoGeniC UCB Bandit...")

    bank = HypothesisBank()

    # Simulate 100 trials
    for i in range(100):
        # Select hypothesis
        hyp = bank.select_hypothesis()

        # Simulate reward (some hypotheses are better than others)
        if "warm lighting" in hyp.text:
            reward = np.random.beta(8, 2)  # Usually good
        elif "dramatic shadows" in hyp.text:
            reward = np.random.beta(6, 4)  # Mixed results
        else:
            reward = np.random.beta(5, 5)  # Average

        success = reward > 0.6

        bank.update_hypothesis(hyp.hypothesis_id, reward, success)

        if (i + 1) % 20 == 0:
            print(f"\nAfter {i + 1} trials:")
            top3 = bank.get_best_hypotheses(top_k=3)
            for h in top3:
                print(
                    f"  - {h.text[:40]:40s} | {h.avg_reward:.3f} ({h.num_trials} trials)"
                )

    stats = bank.get_statistics()
    print(f"   Total trials: {stats['total_trials']}")
    for cat, cat_stats in stats["by_category"].items():
        print(
            f"   {cat:12s}: {cat_stats['count']} hypotheses, "
            f"{cat_stats['avg_success_rate']:.2f} success rate"
        )

    # Save
    bank.save("/tmp/hypothesis_bank.json")

    # Load
    bank2 = HypothesisBank.load("/tmp/hypothesis_bank.json")
