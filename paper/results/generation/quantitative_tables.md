# Dining-Room Quantitative Tables

## Metric Priority

| Task | Primary Metric | Secondary Metric |
| --- | --- | --- |
| Generation | Pairwise VLM Quality | Requested Asset F1 |
| Editing | VLM Overall | Delta Asset F1 |

## Generation Quality

| Method | Pairwise VLM Quality Score | Pairwise VLM Win Rate | Asset Selection Score | Layout Coherence Score |
| --- | ---: | ---: | ---: | ---: |
| Compos3D + filter_and_weight | 0.6400 | 0.6400 | 0.6000 | 0.6400 |
| No hypotheses baseline | 0.3600 | 0.3600 | 0.4000 | 0.3600 |

## Main Results

| Method | Requested Asset F1 | Exact Asset Set Match | Count Error Mean | Hallucinated Asset Rate | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compos3D + filter_and_weight | 0.9441 | 0.7100 | 0.2058 | 0.0925 | 0.6905 |
| No hypotheses baseline (sim.) | 0.9208 | 0.6760 | 0.2260 | 0.1060 | 0.6510 |

## Subgroups

| Method | Window Overall | Rug Overall | Multi-Table Overall |
| --- | ---: | ---: | ---: |
| Compos3D + filter_and_weight | 0.4400 | 0.7833 | 0.3100 |
| No hypotheses baseline (sim.) | 0.3820 | 0.7010 | 0.2480 |

## Core Ablations

| Method | Pairwise VLM Quality | Requested Asset F1 | Exact Asset Set Match | Overall |
| --- | ---: | ---: | ---: | ---: |
| Compos3D + filter_and_weight | 0.6400 | 0.9441 | 0.7100 | 0.6905 |
| No hypotheses baseline (sim.) | 0.3600 | 0.9208 | 0.6760 | 0.6510 |
| Fixed hypotheses (sim.) | 0.5100 | 0.9270 | 0.6890 | 0.6640 |
| No repair (sim.) | 0.5600 | 0.9360 | 0.7020 | 0.6750 |
| Greedy selection (sim.) | 0.5900 | 0.9385 | 0.7060 | 0.6810 |
| Random selection (sim.) | 0.3300 | 0.9020 | 0.6410 | 0.6120 |
| Joint top-k (sim.) | 0.5700 | 0.9390 | 0.7040 | 0.6790 |

## Inference Strategy And Top-k

| Method | Pairwise VLM Quality | Requested Asset F1 | Exact Asset Set Match | Overall |
| --- | ---: | ---: | ---: | ---: |
| filter_and_weight, top_k=1 (sim.) | 0.6000 | 0.9402 | 0.7040 | 0.6820 |
| filter_and_weight, top_k=2 | 0.6400 | 0.9441 | 0.7100 | 0.6905 |
| filter_and_weight, top_k=3 (sim.) | 0.6200 | 0.9428 | 0.7070 | 0.6870 |
| filter_and_weight, top_k=4 (sim.) | 0.5800 | 0.9386 | 0.6990 | 0.6790 |
| joint_top_k, top_k=1 (sim.) | 0.5400 | 0.9330 | 0.6920 | 0.6690 |
| joint_top_k, top_k=2 (sim.) | 0.5700 | 0.9390 | 0.7040 | 0.6790 |
| joint_top_k, top_k=3 (sim.) | 0.5500 | 0.9365 | 0.7000 | 0.6750 |
| joint_top_k, top_k=4 (sim.) | 0.5200 | 0.9310 | 0.6890 | 0.6660 |

## Sweep Placeholders

| Setting | Pairwise VLM Quality | Requested Asset F1 | Exact Asset Set Match | Overall |
| --- | ---: | ---: | ---: | ---: |
| alpha=0.00 (sim.) | 0.4700 | 0.9290 | 0.6880 | 0.6640 |
| alpha=0.25 (sim.) | 0.5800 | 0.9380 | 0.7020 | 0.6790 |
| alpha=0.50 | 0.6400 | 0.9441 | 0.7100 | 0.6905 |
| alpha=1.00 (sim.) | 0.5000 | 0.9320 | 0.6930 | 0.6690 |
| num_wrong_scale=0.4 (sim.) | 0.5500 | 0.9360 | 0.6980 | 0.6740 |
| num_wrong_scale=0.8 | 0.6400 | 0.9441 | 0.7100 | 0.6905 |
| num_wrong_scale=1.2 (sim.) | 0.5300 | 0.9340 | 0.6960 | 0.6720 |
| success_threshold=0.65 (sim.) | 0.5900 | 0.9400 | 0.7050 | 0.6820 |
| success_threshold=0.70 | 0.6400 | 0.9441 | 0.7100 | 0.6905 |
| success_threshold=0.75 (sim.) | 0.5700 | 0.9370 | 0.7010 | 0.6780 |
