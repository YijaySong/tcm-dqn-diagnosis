# V1 消融实验

## 最简单的操作方式

先打开本目录的 `check_gpu.py`，点击 VS Code 右上角运行，确认输出
`CUDA 是否可用: True`。然后：

不需要输入命令。进入想研究的子文件夹，打开其中的 `run.py`，点击 VS Code
右上角“运行 Python 文件”。脚本会自动运行 3 个随机种子、汇总结果并在同一
子文件夹生成：

- `latest_comparison.png`：两个关键指标的横向对比图，绿色柱为最优方案；
- `latest_result.txt`：结果目录和最优方案的文字摘要。

默认是正式实验：3 个随机种子、最多 50 个 RL epoch。一次完整实验可能需要数小时，
运行期间不要关闭 VS Code 终端。所有运行器都从结构化 `test_metrics.json` 读取结果，
并以 `validation_metrics.json` 的跨 seed 指标选择方案；test 只作最终报告。不同实验共享症状组隔离划分、验证集选模和最终测试协议。一键消融不保存体积较大的
`last.pt`（消融不需要恢复训练），但会保留用于验证最优模型的 `best.pt` 和全部指标。

在 Windows NVIDIA GPU 上，一键消融会自动传入 `device=auto`、启用 CUDA AMP 和
TF32。查看终端或每个 seed 的 `train.log`：出现 `device=cuda`、GPU 名称和
`AMP=True` 才代表 GPU 加速生效；若显示 `device=cpu`，需要检查 NVIDIA 驱动及
当前 Python 环境是否安装了 CUDA 版 PyTorch。单 GPU 会顺序执行各方案，避免并行
进程争抢显存；CUDA 内部已自动并行计算，无需手工设置“GPU 多线程”。

## 命令行方式（可选）

```bash
python3 syj/lhz/mymodel_v1/ablations/run_family.py supervision_components --dry-run -- --episodes 2 --pretrain-epochs 1
python3 syj/lhz/mymodel_v1/ablations/run_family.py supervision_components --seeds 9,17,29 -- --episodes 50
```

`--` 后的参数为所有变体共享的预算或数据参数；各变体自身的控制开关会在命令末尾写入，不能被误覆盖。每个批次输出 `manifest.json`、`per_seed_test_metrics.csv` 和 `cross_seed_test_summary.csv`，以及各 `experiment/seed_N/` 自己的模型、划分和 JSON。

| family | 只研究的问题 | 变体 |
|---|---|---|
| `supervision_components` | 预训练、专家 warmup、辅助 BCE 和 TD/RL 的贡献 | full hybrid、去预训练、去 warmup、去辅助 BCE、pretrain+RL、supervised only、pure RL |
| `reward_simplification` | 最终加权 F-beta 奖励的组成 | 最终无步成本方案、重新加入步成本、去掉类别权重 |
| `network_exploration` | epsilon 与 NoisyNet 的必要性 | epsilon only、noisy only、无探索 |
| `tail_supervised_loss` | 监督端长尾处理 | plain BCE、class-balanced BCE、focal BCE；固定关闭 rarity-PER |
| `tail_rare_per` | 回放端长尾处理 | 关闭/开启持续 rarity-PER；继承批次冻结的监督损失 |
| `stop_policy` | 最小输出数量对停止校准的影响 | min actions=0/1/2 |

长尾的两个实验族刻意拆开，避免把监督损失和 PER 同时改变。各类别的研究问题和预期控制变量见对应目录的 README。

`tail_strategy/run.py` 只是依次运行 `tail_supervised_loss` 与 `tail_rare_per` 的快捷入口，不是第三个组合消融。两个子实验已完成，不要再运行它。

六组清洗数据消融已完成，并同步到 `final_config.py` v8。旧结果仅用于历史审计。
各正式批次的数据路径、数据 SHA-256、主配置、变体和训练源码指纹均保存在
`batch_protocol.json` 中。详细选择理由见各子目录 README。
