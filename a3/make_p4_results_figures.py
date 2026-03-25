import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, sys

OUT = "a3/latex/figures"
os.makedirs(OUT, exist_ok=True)


TASKS = [
    "agi_eval_lsat_ar",
    "arc_challenge",
    "arc_easy",
    "bigbench_cs_algorithms",
    "bigbench_dyck_languages",
    "bigbench_language_identification",
    "bigbench_operators",
    "bigbench_qa_wikidata",
    "bigbench_repeat_copy_logic",
    "boolq",
    "commonsense_qa",
    "copa",
    "coqa",
    "hellaswag",
    "hellaswag_zeroshot",
    "jeopardy",
    "lambada_openai",
    "openbook_qa",
    "piqa",
    "squad",
    "winograd",
    "winogrande",
]

TASK_SHORT = {
    "agi_eval_lsat_ar": "lsat_ar",
    "arc_challenge": "arc_chal",
    "arc_easy": "arc_easy",
    "bigbench_cs_algorithms": "bb_cs",
    "bigbench_dyck_languages": "bb_dyck",
    "bigbench_language_identification": "bb_lang",
    "bigbench_operators": "bb_ops",
    "bigbench_qa_wikidata": "bb_wiki",
    "bigbench_repeat_copy_logic": "bb_rcl",
    "boolq": "boolq",
    "commonsense_qa": "csqa",
    "copa": "copa",
    "coqa": "coqa",
    "hellaswag": "hella",
    "hellaswag_zeroshot": "hella_0s",
    "jeopardy": "jeopardy",
    "lambada_openai": "lambada",
    "openbook_qa": "obqa",
    "piqa": "piqa",
    "squad": "squad",
    "winograd": "winograd",
    "winogrande": "winogrande",
}


DATA = {
    "d8 baseline\n(41.9M)": dict(
        zip(
            TASKS,
            [
                0.0652,
                0.0187,
                0.2987,
                0.4240,
                0.0080,
                0.1705,
                0.0952,
                0.0900,
                0.0000,
                -0.3579,
                0.0350,
                0.0000,
                0.0920,
                0.0507,
                0.0533,
                0.0000,
                0.2100,
                0.0213,
                0.2360,
                0.1040,
                0.1282,
                0.0680,
            ],
        )
    ),
    "d12 baseline\n(110.1M)": dict(
        zip(
            TASKS,
            [
                0.0272,
                0.0453,
                0.3813,
                0.4160,
                0.0420,
                0.1749,
                0.1000,
                0.2860,
                0.0000,
                -0.3789,
                0.1725,
                0.0800,
                0.1400,
                0.1547,
                0.1627,
                0.0060,
                0.2940,
                0.1040,
                0.3680,
                0.1740,
                0.1209,
                0.0320,
            ],
        )
    ),
    "d16 baseline\n(234.9M)": dict(
        zip(
            TASKS,
            [
                0.0326,
                0.1387,
                0.5307,
                0.4160,
                0.0700,
                0.1991,
                0.1524,
                0.3740,
                0.0312,
                -0.2842,
                0.1150,
                0.1400,
                0.2200,
                0.2480,
                0.2427,
                0.0440,
                0.3380,
                0.1547,
                0.4240,
                0.2960,
                0.1722,
                0.0360,
            ],
        )
    ),
    "d20 baseline\n(435.2M)": dict(
        zip(
            TASKS,
            [
                0.0598,
                0.1893,
                0.5947,
                0.4540,
                0.0980,
                0.2013,
                0.1714,
                0.4400,
                0.0312,
                -0.0632,
                0.0725,
                0.2800,
                0.2480,
                0.3280,
                0.3520,
                0.0720,
                0.3800,
                0.1733,
                0.5240,
                0.4180,
                0.2601,
                0.1280,
            ],
        )
    ),
    "d20 MTP-2\n+ curriculum\n(435.2M)": dict(
        zip(
            TASKS,
            [
                0.0543,
                0.2027,
                0.6080,
                0.0680,
                0.0660,
                0.2035,
                0.0143,
                0.0700,
                0.0000,
                -0.3368,
                0.0250,
                0.3200,
                0.2260,
                0.3333,
                0.3253,
                0.0180,
                0.3420,
                0.1760,
                0.5040,
                0.2580,
                0.3187,
                0.0760,
            ],
        )
    ),
}

AGGREGATE = {
    "d8 baseline\n(41.9M)": 0.0823,
    "d12 baseline\n(110.1M)": 0.1319,
    "d16 baseline\n(234.9M)": 0.1860,
    "d20 baseline\n(435.2M)": 0.2460,
    "d20 MTP-2\n+ curriculum\n(435.2M)": 0.1760,
}

VAL_BPB = {
    "d8 baseline\n(41.9M)": 0.9724,
    "d12 baseline\n(110.1M)": 0.8626,
    "d16 baseline\n(234.9M)": 0.7933,
    "d20 baseline\n(435.2M)": 0.7445,
    "d20 MTP-2\n+ curriculum\n(435.2M)": 0.9285,
}

MODELS = list(AGGREGATE.keys())
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1, ax2 = axes
for i, (model, core) in enumerate(AGGREGATE.items()):
    bar = ax1.bar(i, core, color=COLORS[i], width=0.6, label=model)
    ax1.text(i, core + 0.004, f"{core:.4f}", ha="center", va="bottom", fontsize=8)

ax1.axhline(
    0.2565, linestyle="--", color="gray", linewidth=1.2, label="GPT-2 ref (0.2565)"
)
ax1.set_xticks(range(len(MODELS)))
ax1.set_xticklabels(
    [m.replace("\n", " ") for m in MODELS], rotation=15, ha="right", fontsize=8
)
ax1.set_ylabel("CORE Aggregate (↑)", fontsize=10)
ax1.set_title("CORE Aggregate by Model", fontsize=11)
ax1.set_ylim(0, 0.32)
ax1.legend(fontsize=7)
ax1.grid(axis="y", linestyle="--", alpha=0.4)


for i, (model, bpb) in enumerate(VAL_BPB.items()):
    ax2.bar(i, bpb, color=COLORS[i], width=0.6)
    ax2.text(i, bpb + 0.005, f"{bpb:.4f}", ha="center", va="bottom", fontsize=8)

ax2.set_xticks(range(len(MODELS)))
ax2.set_xticklabels(
    [m.replace("\n", " ") for m in MODELS], rotation=15, ha="right", fontsize=8
)
ax2.set_ylabel("Validation BPB (↓)", fontsize=10)
ax2.set_title("Validation BPB by Model", fontsize=11)
ax2.set_ylim(0.6, 1.05)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Part 4: Model Comparison Summary", fontsize=12, fontweight="bold")
fig.tight_layout()
path1 = os.path.join(OUT, "p4_core_aggregate.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path1}")


KEY_MODELS = [
    "d16 baseline\n(234.9M)",
    "d20 baseline\n(435.2M)",
    "d20 MTP-2\n+ curriculum\n(435.2M)",
]
KEY_COLORS = [COLORS[2], COLORS[3], COLORS[4]]

fig2, ax = plt.subplots(figsize=(22, 5))
x = np.arange(len(TASKS))
W = 0.22
for j, (model, col) in enumerate(zip(KEY_MODELS, KEY_COLORS)):
    vals = [DATA[model][t] for t in TASKS]
    ax.bar(
        x + j * W - W, vals, W, label=model.replace("\n", " "), color=col, alpha=0.85
    )

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([TASK_SHORT[t] for t in TASKS], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Centered Accuracy", fontsize=10)
ax.set_title(
    "Per-Task CORE: d16 baseline vs d20 baseline vs d20 MTP-2+curriculum", fontsize=11
)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig2.tight_layout()
path2 = os.path.join(OUT, "p4_core_pertask.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {path2}")


try:
    import wandb

    api = wandb.Api(timeout=60)
    entity = "rishit_dagli"

    runs = api.runs(
        f"{entity}/nanochat-part4", filters={"display_name": "p4_d20_mtp2_phase2"}
    )
    if not runs:
        raise ValueError("p4_d20_mtp2_phase2 not found")
    run = runs[0]

    history = run.scan_history()
    steps, losses, cores = [], [], []
    for row in history:
        s = row.get("step") or row.get("_step")
        l = row.get("train/loss")
        c = row.get("core_metric")
        if s is not None and l is not None:
            steps.append(s)
            losses.append(l)
        if s is not None and c is not None:
            cores.append((s, c))

    fig3, (ax_l, ax_c) = plt.subplots(1, 2, figsize=(12, 4))

    if steps:
        ax_l.plot(steps, losses, color="#4C72B0", lw=1.5, alpha=0.8)
        ax_l.set_xlabel("Step", fontsize=10)
        ax_l.set_ylabel("Training Loss", fontsize=10)
        ax_l.set_title("d20 MTP-2+curriculum — Training Loss (Phase 2)", fontsize=10)
        ax_l.grid(linestyle="--", alpha=0.4)
        ax_l.axvline(
            3669,
            color="orange",
            linestyle="--",
            linewidth=1.2,
            label="Phase 1→2 boundary",
        )
        ax_l.legend(fontsize=8)

    if cores:
        csteps, cvals = zip(*cores)
        ax_c.plot(csteps, cvals, color="#C44E52", lw=1.5, marker="o", markersize=4)
        ax_c.set_xlabel("Step", fontsize=10)
        ax_c.set_ylabel("CORE Metric", fontsize=10)
        ax_c.set_title(
            "d20 MTP-2+curriculum — CORE Metric During Training", fontsize=10
        )
        ax_c.grid(linestyle="--", alpha=0.4)

        ax_c.axhline(
            0.2460,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label="d20 baseline CORE (0.246)",
        )
        ax_c.axhline(
            0.1860,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="d16 baseline CORE (0.186)",
        )
        ax_c.legend(fontsize=8)

    fig3.suptitle(
        "d20 MTP-2 + Curriculum: Training Dynamics", fontsize=12, fontweight="bold"
    )
    fig3.tight_layout()
    path3 = os.path.join(OUT, "p4_training_curves.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {path3}")
except Exception as e:
    print(f"[W&B training curves skipped: {e}]")


print("\n=== Model Comparison Summary ===")
print(f"{'Model':<38} {'BPB':>8} {'CORE':>8}")
print("-" * 56)
for m in MODELS:
    print(f"{m.replace(chr(10), ' '):<38} {VAL_BPB[m]:>8.4f} {AGGREGATE[m]:>8.4f}")
