"""
Data is taken from /tmp/a2mtp_eval_report.log (step 4692, d16).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np



CONFIGS = ["Baseline", "MTP-2", "MTP-4", "MTP-2+YaRN"]
CONFIG_COLORS = plt.cm.tab10([0, 1, 2, 3])   

CORE_AGGREGATE = {
    "Baseline":   0.185954,
    "MTP-2":      0.139190,
    "MTP-4":      0.142791,
    "MTP-2+YaRN": 0.144624,
}

BPB = {
    "Baseline":   0.794471,
    "MTP-2":      0.971154,
    "MTP-4":      0.977287,
    "MTP-2+YaRN": 0.971859,
}


PER_TASK = {
    "agi_eval_lsat_ar":              {"Baseline":  0.0326, "MTP-2":  0.0489, "MTP-4":  0.0924, "MTP-2+YaRN":  0.1033},
    "arc_challenge":                 {"Baseline":  0.1387, "MTP-2":  0.1173, "MTP-4":  0.1067, "MTP-2+YaRN":  0.1120},
    "arc_easy":                      {"Baseline":  0.5307, "MTP-2":  0.5333, "MTP-4":  0.5387, "MTP-2+YaRN":  0.5120},
    "bigbench_cs_algorithms":        {"Baseline":  0.4160, "MTP-2":  0.3200, "MTP-4":  0.3800, "MTP-2+YaRN":  0.1900},
    "bigbench_dyck_languages":       {"Baseline":  0.0700, "MTP-2":  0.0440, "MTP-4":  0.0100, "MTP-2+YaRN":  0.0700},
    "bigbench_language_identification": {"Baseline": 0.1991, "MTP-2": 0.1793, "MTP-4": 0.1837, "MTP-2+YaRN": 0.1573},
    "bigbench_operators":            {"Baseline":  0.1524, "MTP-2":  0.0190, "MTP-4":  0.0524, "MTP-2+YaRN":  0.0190},
    "bigbench_qa_wikidata":          {"Baseline":  0.3740, "MTP-2":  0.0540, "MTP-4":  0.1120, "MTP-2+YaRN":  0.0880},
    "bigbench_repeat_copy_logic":    {"Baseline":  0.0312, "MTP-2":  0.0000, "MTP-4":  0.0000, "MTP-2+YaRN":  0.0000},
    "boolq":                         {"Baseline": -0.2842, "MTP-2": -0.4737, "MTP-4": -0.3158, "MTP-2+YaRN": -0.3211},
    "commonsense_qa":                {"Baseline":  0.1150, "MTP-2":  0.0425, "MTP-4":  0.0425, "MTP-2+YaRN":  0.1150},
    "copa":                          {"Baseline":  0.1400, "MTP-2":  0.3000, "MTP-4":  0.1200, "MTP-2+YaRN":  0.2400},
    "coqa":                          {"Baseline":  0.2200, "MTP-2":  0.1280, "MTP-4":  0.1920, "MTP-2+YaRN":  0.1600},
    "hellaswag":                     {"Baseline":  0.2480, "MTP-2":  0.2347, "MTP-4":  0.2427, "MTP-2+YaRN":  0.2320},
    "hellaswag_zeroshot":            {"Baseline":  0.2427, "MTP-2":  0.2347, "MTP-4":  0.2000, "MTP-2+YaRN":  0.2587},
    "jeopardy":                      {"Baseline":  0.0440, "MTP-2":  0.0020, "MTP-4":  0.0040, "MTP-2+YaRN":  0.0060},
    "lambada_openai":                {"Baseline":  0.3380, "MTP-2":  0.3000, "MTP-4":  0.2960, "MTP-2+YaRN":  0.2760},
    "openbook_qa":                   {"Baseline":  0.1547, "MTP-2":  0.1440, "MTP-4":  0.1493, "MTP-2+YaRN":  0.1413},
    "piqa":                          {"Baseline":  0.4240, "MTP-2":  0.4000, "MTP-4":  0.4080, "MTP-2+YaRN":  0.4440},
    "squad":                         {"Baseline":  0.2960, "MTP-2":  0.1840, "MTP-4":  0.2240, "MTP-2+YaRN":  0.1780},
    "winograd":                      {"Baseline":  0.1722, "MTP-2":  0.2381, "MTP-4":  0.1429, "MTP-2+YaRN":  0.1722},
    "winogrande":                    {"Baseline":  0.0360, "MTP-2":  0.0120, "MTP-4": -0.0400, "MTP-2+YaRN":  0.0280},
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "latex", "figures")
os.makedirs(OUT_DIR, exist_ok=True)



fig1, ax1 = plt.subplots(figsize=(7, 5))

x = np.arange(len(CONFIGS))
bars = ax1.bar(x, [CORE_AGGREGATE[c] for c in CONFIGS],
               color=CONFIG_COLORS, edgecolor="white", linewidth=0.5,
               width=0.55, zorder=3)


for bar, cfg in zip(bars, CONFIGS):
    v = CORE_AGGREGATE[cfg]
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
             f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(CONFIGS, fontsize=11)
ax1.set_ylabel("CORE Aggregate Score ↑", fontsize=11)
ax1.set_ylim(0, max(CORE_AGGREGATE.values()) * 1.18)
ax1.set_title("Part 2 — CORE Aggregate Score by Config (d16, step 4692)", fontsize=11)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.axhline(CORE_AGGREGATE["Baseline"], color="grey", linewidth=1,
            linestyle="--", alpha=0.6, label="Baseline")

fig1.tight_layout()
p1 = os.path.join(OUT_DIR, "p2_core_bar.png")
fig1.savefig(p1, dpi=180, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {p1}")



tasks = list(PER_TASK.keys())           
n_tasks  = len(tasks)
n_cfgs   = len(CONFIGS)
BAR_W    = 0.18
offsets  = np.linspace(-(n_cfgs - 1) / 2 * BAR_W,
                        (n_cfgs - 1) / 2 * BAR_W, n_cfgs)
x_base   = np.arange(n_tasks)

fig2, ax2 = plt.subplots(figsize=(26, 6))
for i, cfg in enumerate(CONFIGS):
    vals = [PER_TASK[t][cfg] for t in tasks]
    ax2.bar(x_base + offsets[i], vals, width=BAR_W,
            color=CONFIG_COLORS[i], label=cfg,
            edgecolor="white", linewidth=0.3, zorder=3)

ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xticks(x_base)
ax2.set_xticklabels([t.replace("_", "\n") for t in tasks],
                    fontsize=6.5, rotation=0, ha="center")
ax2.set_ylabel("Centred Score ↑", fontsize=10)
ax2.set_title("Part 2 — CORE Per-Task Centred Scores (d16, step 4692)", fontsize=11)
ax2.grid(axis="y", alpha=0.25, zorder=0)
ax2.legend(fontsize=9, loc="upper right",
           title="Config", title_fontsize=8)

fig2.tight_layout()
p2 = os.path.join(OUT_DIR, "p2_core_per_task.png")
fig2.savefig(p2, dpi=160, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {p2}")



COL_W = 13
print()
print(f"{'Task':<36}" + "".join(f"{c:>{COL_W}}" for c in CONFIGS))
print("-" * (36 + COL_W * n_cfgs))
for t in tasks:
    row = f"{t:<36}"
    for cfg in CONFIGS:
        row += f"{PER_TASK[t][cfg]:>{COL_W}.4f}"
    print(row)
print("-" * (36 + COL_W * n_cfgs))
agg_row = f"{'CORE aggregate':<36}"
for cfg in CONFIGS:
    agg_row += f"{CORE_AGGREGATE[cfg]:>{COL_W}.4f}"
print(agg_row)
print()
bpb_row = f"{'Val BPB':<36}"
for cfg in CONFIGS:
    bpb_row += f"{BPB[cfg]:>{COL_W}.6f}"
print(bpb_row)
print()
print("Done.")
