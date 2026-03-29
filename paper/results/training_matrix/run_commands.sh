#!/usr/bin/env bash

set -euo pipefail



source api_key

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_full_system --config-path paper/results/training_matrix/configs/full_system.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_full_system_val --method-name full_system --config-path paper/results/training_matrix/configs/full_system.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_full_system/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_no_hypotheses --config-path paper/results/training_matrix/configs/no_hypotheses.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_no_hypotheses_val --method-name no_hypotheses --config-path paper/results/training_matrix/configs/no_hypotheses.json --inference-strategy filter_and_weight --render-scene --use-empty-bank

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_fixed_hypotheses --config-path paper/results/training_matrix/configs/fixed_hypotheses.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_fixed_hypotheses_val --method-name fixed_hypotheses --config-path paper/results/training_matrix/configs/fixed_hypotheses.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_fixed_hypotheses/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_no_repair --config-path paper/results/training_matrix/configs/no_repair.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_no_repair_val --method-name no_repair --config-path paper/results/training_matrix/configs/no_repair.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_no_repair/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_greedy --config-path paper/results/training_matrix/configs/greedy.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_greedy_val --method-name greedy --config-path paper/results/training_matrix/configs/greedy.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_greedy/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_random --config-path paper/results/training_matrix/configs/random.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_random_val --method-name random --config-path paper/results/training_matrix/configs/random.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_random/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_alpha_0_0 --config-path paper/results/training_matrix/configs/alpha_0_0.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_alpha_0_0_val --method-name alpha_0_0 --config-path paper/results/training_matrix/configs/alpha_0_0.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_alpha_0_0/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_alpha_0_25 --config-path paper/results/training_matrix/configs/alpha_0_25.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_alpha_0_25_val --method-name alpha_0_25 --config-path paper/results/training_matrix/configs/alpha_0_25.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_alpha_0_25/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_alpha_0_5 --config-path paper/results/training_matrix/configs/alpha_0_5.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_alpha_0_5_val --method-name alpha_0_5 --config-path paper/results/training_matrix/configs/alpha_0_5.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_alpha_0_5/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_alpha_1_0 --config-path paper/results/training_matrix/configs/alpha_1_0.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_alpha_1_0_val --method-name alpha_1_0 --config-path paper/results/training_matrix/configs/alpha_1_0.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_alpha_1_0/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_num_wrong_scale_0_4 --config-path paper/results/training_matrix/configs/num_wrong_scale_0_4.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_num_wrong_scale_0_4_val --method-name num_wrong_scale_0_4 --config-path paper/results/training_matrix/configs/num_wrong_scale_0_4.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_num_wrong_scale_0_4/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_num_wrong_scale_0_8 --config-path paper/results/training_matrix/configs/num_wrong_scale_0_8.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_num_wrong_scale_0_8_val --method-name num_wrong_scale_0_8 --config-path paper/results/training_matrix/configs/num_wrong_scale_0_8.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_num_wrong_scale_0_8/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_num_wrong_scale_1_2 --config-path paper/results/training_matrix/configs/num_wrong_scale_1_2.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_num_wrong_scale_1_2_val --method-name num_wrong_scale_1_2 --config-path paper/results/training_matrix/configs/num_wrong_scale_1_2.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_num_wrong_scale_1_2/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_success_threshold_0_65 --config-path paper/results/training_matrix/configs/success_threshold_0_65.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_success_threshold_0_65_val --method-name success_threshold_0_65 --config-path paper/results/training_matrix/configs/success_threshold_0_65.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_success_threshold_0_65/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_success_threshold_0_70 --config-path paper/results/training_matrix/configs/success_threshold_0_70.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_success_threshold_0_70_val --method-name success_threshold_0_70 --config-path paper/results/training_matrix/configs/success_threshold_0_70.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_success_threshold_0_70/hypothesis_bank.json

./.venv/bin/compos3d train-hypotheses --dataset-path examples/vertical_slice_dataset.json --output-dir artifacts/training --experiment-name paper_dining_success_threshold_0_75 --config-path paper/results/training_matrix/configs/success_threshold_0_75.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_val.json --output-dir paper/results/generation/paper_dining_success_threshold_0_75_val --method-name success_threshold_0_75 --config-path paper/results/training_matrix/configs/success_threshold_0_75.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/paper_dining_success_threshold_0_75/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/filter_and_weight_topk_1 --method-name filter_and_weight_topk_1 --config-path paper/results/training_matrix/configs/filter_and_weight_topk_1.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/filter_and_weight_topk_2 --method-name filter_and_weight_topk_2 --config-path paper/results/training_matrix/configs/filter_and_weight_topk_2.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/filter_and_weight_topk_3 --method-name filter_and_weight_topk_3 --config-path paper/results/training_matrix/configs/filter_and_weight_topk_3.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/filter_and_weight_topk_4 --method-name filter_and_weight_topk_4 --config-path paper/results/training_matrix/configs/filter_and_weight_topk_4.json --inference-strategy filter_and_weight --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/joint_top_k_topk_1 --method-name joint_top_k_topk_1 --config-path paper/results/training_matrix/configs/joint_top_k_topk_1.json --inference-strategy joint_top_k --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/joint_top_k_topk_2 --method-name joint_top_k_topk_2 --config-path paper/results/training_matrix/configs/joint_top_k_topk_2.json --inference-strategy joint_top_k --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/joint_top_k_topk_3 --method-name joint_top_k_topk_3 --config-path paper/results/training_matrix/configs/joint_top_k_topk_3.json --inference-strategy joint_top_k --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json

./.venv/bin/python scripts/run_paper_generation_eval.py --benchmark-path paper/benchmarks/dining_test.json --output-dir paper/results/generation/joint_top_k_topk_4 --method-name joint_top_k_topk_4 --config-path paper/results/training_matrix/configs/joint_top_k_topk_4.json --inference-strategy joint_top_k --render-scene --bank-path artifacts/training/claude_qwen/hypothesis_bank.json
