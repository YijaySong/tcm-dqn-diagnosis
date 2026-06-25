#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

run_exp() {
  local name="$1"
  shift
  echo "=== Running: $name ==="
  "$@" 2>&1 | tee "logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo
}

# 1) 纯监督 baseline
run_exp "supervised_baseline" python3 syj/exp_supervised_baseline.py --mode top2

# 2) 无RL消融：纯监督阈值模式
run_exp "no_rl" python3 syj/exp_no_rl.py

# 3) 无预训练
run_exp "no_pretrain" python3 syj/exp_no_pretrain.py -episode 50 -batch 64 -mem_capacity 10000 -nn_units 128 -nn_units2 64 -lr 0.001 -patience 10

# 4) 无专家轨迹
run_exp "no_expert_trajectory" python3 syj/exp_no_expert_trajectory.py -episode 50 -batch 64 -mem_capacity 10000 -nn_units 128 -nn_units2 64 -lr 0.001 -patience 10

# 5) 论文更安全版（关闭预训练和专家轨迹，保留RL主流程）
run_exp "safe_dqn" python3 syj/differentiation_safe.py -episode 50 -batch 64 -mem_capacity 10000 -nn_units 128 -nn_units2 64 -lr 0.001 -patience 10

echo "All ablation runs finished. Logs are under: $ROOT_DIR/logs"
