# LHZ 消融实验

本目录保存 LHZ 模块的消融实验入口脚本。所有实验都复用 `syj/lhz/mymodel/main.py`，只改变关键实验变量，不修改模型结构、奖励函数、数据划分或评估指标。

## 实验列表

| 脚本 | 监督预训练 | 专家轨迹 ReplayMemory 预填充 | 后续 RL 训练 | 论文表格建议名称 | 目的 |
|---|---:|---:|---:|---|---|
| `run_full.py` | 是 | 是 | 是 | Full | 完整模型，对照组 |
| `run_no_pretrain.py` | 否 | 是 | 是 | w/o Pretrain | 验证监督预训练的贡献 |
| `run_pretrain_only.py` | 是 | 否 | 否 | Pretrain Only | 验证只有监督预训练、没有后续 RL 的效果 |
| `run_no_expert_warmup.py` | 是 | 否 | 是 | w/o Expert Warmup | 验证专家轨迹预填充 ReplayMemory 的贡献 |
| `run_rl_only.py` | 否 | 否 | 是 | RL Only | 验证无预训练、无专家轨迹时纯 RL 的效果 |

## 一键运行并汇总指标

在项目根目录运行全部实验：

```bash
python3 syj/lhz/ablations/run_all.py
```

运行完成后会自动展示：

- 汇总指标：每个实验 × 每种评估方式。
- 逐标签指标：每个证候要素的 Precision / Recall / F1 / Accuracy / Support / PredCount。

结果会保存到：

```text
syj/lhz/ablations/results/<时间戳>/
├── all_metrics.json
├── aggregate_metrics.csv
├── label_metrics.csv
├── full/
├── no_pretrain/
├── pretrain_only/
├── no_expert_warmup/
└── rl_only/
```

其中：

- `aggregate_metrics.csv`：适合放入论文消融表的汇总指标。
- `label_metrics.csv`：逐证候要素指标。
- `all_metrics.json`：完整机器可读结果。
- 每个实验子目录中会保存该实验的日志、模型副本和划分数据副本。

## 可视化指标

运行完 `run_all.py` 后，可以把核心指标画成图、辅助指标整理成表：

```bash
python3 syj/lhz/ablations/plot_ablation_metrics.py
```

默认会读取 `syj/lhz/ablations/results/` 下最新的实验批次，并生成：

```text
syj/lhz/ablations/results/<时间戳>/ablation_metrics.pdf
```

PDF 内容包括：

1. `auto` 自主停止模式的主要指标柱状图。
2. 主要指标热力图。
3. 辅助指标表格。
4. 逐证候要素指标表格。

指定某个实验批次目录：

```bash
python3 syj/lhz/ablations/plot_ablation_metrics.py --result-dir syj/lhz/ablations/results/<时间戳>
```

同时输出 PNG 图片：

```bash
python3 syj/lhz/ablations/plot_ablation_metrics.py --also-png
```

逐标签表默认每个实验展示 support 最高的前 15 个证候要素，可以调整：

```bash
python3 syj/lhz/ablations/plot_ablation_metrics.py --top-n-labels 20
```

## 推荐报告指标

建议在论文消融表中优先报告以下测试集指标：

- `sample_f1`
- `exact_match`
- `micro_f1`
- `macro_f1`

当前评估只输出 `auto` / `测试集-自主停止`：模型自主决定何时停止推荐证候要素。

## 常用命令

只检查将要运行什么，不启动训练：

```bash
python3 syj/lhz/ablations/run_all.py --dry-run
```

为了快速验证流程，只给所有实验追加较少训练轮数：

```bash
python3 syj/lhz/ablations/run_all.py -- -episode 1
```

只运行部分实验：

```bash
python3 syj/lhz/ablations/run_all.py --experiments full no_pretrain pretrain_only
```

隐藏终端中的逐标签长表，但仍保存 `label_metrics.csv`：

```bash
python3 syj/lhz/ablations/run_all.py --hide-labels
```

单独运行某个实验：

```bash
python3 syj/lhz/ablations/run_full.py
python3 syj/lhz/ablations/run_no_pretrain.py
python3 syj/lhz/ablations/run_pretrain_only.py
python3 syj/lhz/ablations/run_no_expert_warmup.py
python3 syj/lhz/ablations/run_rl_only.py
```

单独运行某个实验时，也可以追加主程序参数：

```bash
python3 syj/lhz/ablations/run_full.py -- -episode 5
```

## 安全运行说明

`ablation_utils.py` 会在每个实验运行前备份以下主程序可能覆盖的产物：

- `syj/lhz/mymodel/dqn_model.pth`
- `syj/lhz/mymodel/split_data/`

每个实验结束后，会先将本次实验产生的模型和划分数据复制到该实验自己的结果目录，然后恢复实验开始前的文件状态。因此连续运行多个消融实验时，不会直接覆盖你原先的 LHZ 模型文件。

日志会保存到对应实验结果目录中，因为实验运行时会临时将当前工作目录切换到该目录。

## 注意事项

- `run_pretrain_only.py` 使用 `-episode 0`，因此会执行监督预训练，但跳过后续 RL 训练。
- 完整运行全部实验可能耗时较长；可先使用 `--dry-run` 或 `-- -episode 1` 检查流程。
- 若某个实验失败，默认会停止后续实验；如需继续运行后续实验，可使用 `--continue-on-error`。
