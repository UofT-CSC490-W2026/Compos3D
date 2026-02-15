"""Test HypoGeniC UCB Bandit"""

from compos3d_dp.generation.hypogenic_ucb import HypothesisBank
import tempfile
from pathlib import Path


# Initialize bank
bank = HypothesisBank(exploration_weight=2.0)
print(f"Initialized bank with {len(bank.hypotheses)} hypotheses")

# Test hypothesis selection
hyp = bank.select_hypothesis()
print(f"\nSelected hypothesis: {hyp.text[:50]}...")
print(f"   Category: {hyp.category}")
print(f"   UCB score: {hyp.ucb_score(bank.total_trials):.3f}")

# Test updating hypothesis
bank.update_hypothesis(hyp.hypothesis_id, reward=0.85, success=True)
print("\nUpdated hypothesis")
print(f"   Trials: {hyp.num_trials}")
print(f"   Successes: {hyp.num_successes}")
print(f"   Avg reward: {hyp.avg_reward:.3f}")

# Test category selection
lighting_hyps = bank.select_top_k(k=3, category="lighting")
print("\nSelected top 3 lighting hypotheses:")
for h in lighting_hyps:
    print(f"   - {h.text}")

# Test save/load
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "hypothesis_bank.json"
    bank.save(save_path)
    print(f"\nSaved bank to {save_path}")

    loaded_bank = HypothesisBank.load(save_path)
    print(f"Loaded bank with {len(loaded_bank.hypotheses)} hypotheses")
    assert len(loaded_bank.hypotheses) == len(bank.hypotheses)
    assert loaded_bank.total_trials == bank.total_trials

# Test statistics
stats = bank.get_statistics()
print("\nBank statistics:")
print(f"   Total hypotheses: {stats['total_hypotheses']}")
print(f"   Total trials: {stats['total_trials']}")
for cat, cat_stats in stats["by_category"].items():
    print(f"   {cat}: {cat_stats['count']} hypotheses")

print("\n" + "=" * 60)
