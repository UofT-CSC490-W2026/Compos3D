"""
Generate Part 4 scaling-law figures.
  - p4_scaling_bpb.png   : Val BPB vs. parameter count (log-log), power-law fit + d20 prediction vs actual
  - p4_scaling_core.png  : CORE metric vs. parameter count (semi-log), trend + d20 prediction vs actual

Run from a3/  with:  python3 make_p4_scaling_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── Data ──────────────────────────────────────────────────────────────────────

def scaling_params(d):
    """Non-embedding parameter count for depth-d nanochat (same formula as Part 4 modal)."""
    n = d * 64        # hidden dim
    return d * 12 * n * n + 32768 * n   # 12 × layers × (n_heads attention + MLP)

# Anchor points  (all baseline, Chinchilla-optimal)
DEPTHS     = [8,  12,   16,    20]
PARAMS     = [scaling_params(d) for d in DEPTHS]   # in raw count
PARAMS_M   = [p / 1e6 for p in PARAMS]             # in millions

# Empirical BPB (val/bpb)
BPB_ACTUAL = {
    8:  0.9730,
    12: 0.8631,
    16: 0.7933,
    20: 0.7445,   # Part 3 baseline (ground truth for comparison)
}

# CORE metric
CORE_ACTUAL = {
    8:  0.0714,
    12: 0.1319,
    16: 0.1860,
    20: 0.2460,
}

COLORS = {
    "anchor":     "#4C72B0",
    "fit":        "#4C72B0",
    "predicted":  "#DD8452",
    "actual_d20": "#55A868",
}

output_dir = "a3/latex/figures"
os.makedirs(output_dir, exist_ok=True)

# ── Power-law fit in log-log space (anchor: d8, d12, d16) ─────────────────────

anchor_depths = [8, 12, 16]
x_anchor = np.array([np.log(scaling_params(d)) for d in anchor_depths])
y_bpb    = np.array([np.log(BPB_ACTUAL[d])     for d in anchor_depths])

# Least-squares: log(BPB) = b * log(N) + log(a)
b_bpb, log_a_bpb = np.polyfit(x_anchor, y_bpb, 1)
a_bpb = np.exp(log_a_bpb)

# Predict d20
log_N_d20 = np.log(scaling_params(20))
bpb_pred_d20 = float(np.exp(b_bpb * log_N_d20 + log_a_bpb))

print(f"BPB power-law fit:  BPB = {a_bpb:.4f} × N^({b_bpb:.4f})")
print(f"Predicted d20 BPB : {bpb_pred_d20:.4f}")
print(f"Actual    d20 BPB : {BPB_ACTUAL[20]:.4f}")
print()

# ── CORE trend fit (log N → CORE, simple linear in log-N space) ───────────────

y_core = np.array([CORE_ACTUAL[d] for d in anchor_depths])
# Linear fit: CORE = c1 * log(N) + c0
c1_core, c0_core = np.polyfit(x_anchor, y_core, 1)

core_pred_d20 = float(c1_core * log_N_d20 + c0_core)

print(f"CORE linear fit (in log-N): CORE = {c1_core:.4f}*log(N) + ({c0_core:.4f})")
print(f"Predicted d20 CORE : {core_pred_d20:.4f}")
print(f"Actual    d20 CORE : {CORE_ACTUAL[20]:.4f}")

# ── Figure 1: BPB scaling ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

# Fitted curve over a wide range
N_range = np.logspace(np.log10(30e6), np.log10(600e6), 300)
bpb_fit  = a_bpb * N_range ** b_bpb
ax.plot(N_range / 1e6, bpb_fit, "-", color=COLORS["fit"], lw=2,
        label=rf"Power-law fit:  $\mathrm{{BPB}} = {a_bpb:.3f}\,N^{{{b_bpb:.3f}}}$")

# Anchor points
for d in anchor_depths:
    N_m = scaling_params(d) / 1e6
    ax.scatter([N_m], [BPB_ACTUAL[d]], s=90, zorder=5, color=COLORS["anchor"])
    ax.annotate(f"d{d}", (N_m, BPB_ACTUAL[d]),
                xytext=(4, 4), textcoords="offset points", fontsize=9, color=COLORS["anchor"])

# Predicted d20
N_d20_m = scaling_params(20) / 1e6
ax.scatter([N_d20_m], [bpb_pred_d20], s=120, marker="*", zorder=6,
           color=COLORS["predicted"],
           label=f"Predicted d20:  {bpb_pred_d20:.4f} BPB")

# Actual d20
ax.scatter([N_d20_m], [BPB_ACTUAL[20]], s=90, marker="D", zorder=6,
           color=COLORS["actual_d20"],
           label=f"Actual d20:      {BPB_ACTUAL[20]:.4f} BPB")

# Arrow connecting prediction → actual
ax.annotate("", xy=(N_d20_m, BPB_ACTUAL[20]),
            xytext=(N_d20_m, bpb_pred_d20),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
ax.text(N_d20_m + 6, (bpb_pred_d20 + BPB_ACTUAL[20]) / 2,
        f"Δ={abs(BPB_ACTUAL[20]-bpb_pred_d20):.4f}", fontsize=8, color="gray")

ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
ax.set_xlabel("Non-embedding parameters (log scale)", fontsize=11)
ax.set_ylabel("Validation BPB (log scale)", fontsize=11)
ax.set_title("Scaling Law: Validation BPB vs. Parameter Count", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "p4_scaling_bpb.png"), dpi=150)
plt.close(fig)
print(f"\nSaved: {output_dir}/p4_scaling_bpb.png")

# ── Figure 2: CORE scaling ─────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 5))

# Fitted curve
log_N_range = np.log(N_range)
core_fit = c1_core * log_N_range + c0_core
ax2.plot(N_range / 1e6, core_fit, "-", color=COLORS["fit"], lw=2,
         label=rf"Linear fit (log-N):  $\mathrm{{CORE}} = {c1_core:.3f}\ln N + ({c0_core:.2f})$")

# Anchor points
for d in anchor_depths:
    N_m = scaling_params(d) / 1e6
    ax2.scatter([N_m], [CORE_ACTUAL[d]], s=90, zorder=5, color=COLORS["anchor"])
    ax2.annotate(f"d{d}", (N_m, CORE_ACTUAL[d]),
                 xytext=(4, 4), textcoords="offset points", fontsize=9, color=COLORS["anchor"])

# Predicted d20
ax2.scatter([N_d20_m], [core_pred_d20], s=120, marker="*", zorder=6,
            color=COLORS["predicted"],
            label=f"Predicted d20:  {core_pred_d20:.4f}")

# Actual d20
ax2.scatter([N_d20_m], [CORE_ACTUAL[20]], s=90, marker="D", zorder=6,
            color=COLORS["actual_d20"],
            label=f"Actual d20:      {CORE_ACTUAL[20]:.4f}")

ax2.annotate("", xy=(N_d20_m, CORE_ACTUAL[20]),
             xytext=(N_d20_m, core_pred_d20),
             arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
ax2.text(N_d20_m + 6, (core_pred_d20 + CORE_ACTUAL[20]) / 2,
         f"Δ={abs(CORE_ACTUAL[20]-core_pred_d20):.4f}", fontsize=8, color="gray")

ax2.set_xscale("log")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax2.set_xlabel("Non-embedding parameters (log scale)", fontsize=11)
ax2.set_ylabel("CORE Aggregate Metric (↑ higher is better)", fontsize=11)
ax2.set_title("Scaling Law: CORE Metric vs. Parameter Count", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", linestyle="--", alpha=0.4)

fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, "p4_scaling_core.png"), dpi=150)
plt.close(fig2)
print(f"Saved: {output_dir}/p4_scaling_core.png")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n=== Scaling Law Summary Table ===")
print(f"{'Depth':<6} {'Params (M)':<12} {'BPB actual':<14} {'BPB predicted':<16} {'CORE actual':<14} {'CORE predicted'}")
for d in DEPTHS:
    N_m = scaling_params(d) / 1e6
    log_N = np.log(scaling_params(d))
    bpb_p = float(np.exp(b_bpb * log_N + log_a_bpb))
    core_p = float(c1_core * log_N + c0_core)
    bpb_a  = BPB_ACTUAL.get(d, "—")
    core_a = CORE_ACTUAL.get(d, "—")
    bpb_a_str  = f"{bpb_a:.4f}" if isinstance(bpb_a, float) else bpb_a
    core_a_str = f"{core_a:.4f}" if isinstance(core_a, float) else core_a
    print(f"d{d:<5} {N_m:<12.1f} {bpb_a_str:<14} {bpb_p:<16.4f} {core_a_str:<14} {core_p:.4f}")
print(f"\nPower-law: BPB = {a_bpb:.4f} × N^({b_bpb:.4f})")
print(f"Chinchilla α ≈ {abs(b_bpb):.4f}  (expected ~0.076–0.12 for LLMs)")
