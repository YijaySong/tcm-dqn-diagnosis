# mymodel_v1

`mymodel_v1` 是原 `mymodel` 的独立实验副本。它保留症状集合与已选标签集合的 Dueling DQN 主干，但首先修正了数据协议和测试集使用方式；原目录及其既有结果不会被读取或改写。

## 本版改动

- **数据与泄漏控制**：默认读取清洗记录，再按“规范化症状集合”分组为 70/15/15 的 train/validation/test。症状、标签的词表、Top-N 筛选、类别支持度、奖励权重和长尾分段全部仅从 train 拟合。验证/测试中的训练未见标签会按默认策略丢弃整条样本并记录数量；未见症状保留在 CSV 中、编码为全零并记录 OOV 统计。
- **重要边界**：原始数据没有患者/病例 ID。因此该划分只保证相同症状组合不跨集合，不能声称为患者级独立测试。获得病例 ID 后，应将 `data.py` 的分组键替换为病例 ID 后重新运行全部实验。
- **训练协议**：每个 RL epoch 只在 validation 评估、以 `supported_macro_f1` 选最优 checkpoint；默认关闭早停并完整运行设定的 epoch 数，test 只在恢复最优模型后评估一次。`best.pt` 用于推理，`last.pt` 另存 optimizer、replay 和随机状态用于继续训练。
- **奖励与停止**：清洗数据消融确认使用训练集频率权重下的加权 F1 势函数和唯一 STOP 终止效用，步成本为 0。已移除好奇心、动态 FP/FN、长度和重复稀有标签奖励。STOP 默认 `--min-actions-before-stop 1`，不增加 stop margin；行为采样、DDQN bootstrap 与推理共用合法动作 mask。
- **监督、示范与 RL**：`--use-pretrain`、`--use-expert-warmup`、`--use-aux-bce/--aux-bce-weight` 和 `--run-rl` 相互独立。最终采用专家 Replay warmup、辅助监督权重 0.05 和 RL。专家轨迹会先估算数目并自动扩展 replay，避免预填充后被静默覆盖。
- **网络与损失**：使用 `set_dueling` 且不启用显式 epsilon 或 NoisyNet。监督损失采用类别平衡 BCE。Replay 保留标准 TD-error PER，`rare_per_scale=0`，不再叠加稀有标签 multiplier。

## 运行

### 数据清洗

直接打开 `clean_dataset.py` 并点击 VS Code 右上角运行按钮，会读取原始
`dataset/lhz_data.txt`，在同一目录生成 `lhz_data_cleaned.txt`。原文件不会被覆盖。
同时生成清洗报告、逐项审计 CSV、删除记录 CSV 和低支持度标签删除明细表。

清洗规则：删除明确混入的非症状元数据；删除单标签且症状数超过全数据 P95 的
记录；同症不同证取各标注标签交集，不使用会扩大标签集合的并集；按唯一症状组
计算标签支持度并删除支持度小于5的标签，删除后无标签的记录整条删除。完全重复
记录按当前要求保留，因此会增加对应症状模板在训练损失中的权重；数据划分仍将
相同症状组放在同一个 split，避免重复记录跨 train/validation/test 泄漏。

主程序和可点击消融入口现已默认读取 `dataset/lhz_data_cleaned.txt`。清洗只约束
“症状组合完全相同但结果不同”的记录；结果相同但症状组合不同的数据仍分别保留。
清洗版属于新的实验协议。六组消融已在该协议下完成，旧数据结果只作历史审计。

### 最终配置同步规则

主程序的消融相关默认值统一来自 `final_config.py`。每完成一组正式消融，需同时更新
该文件、对应消融 README 和主 README。直接点击 `main.py` 时会自动采用当前已经确认
的最佳组合，并把配置版本、完整快照和各消融状态写入 `resolved_config.json`。

当前配置版本为 `2026-07-20-v8-cleaned-ablation-final`。六项消融状态均已写入
`final_config.py`；直接点击 `main.py` 会训练该最终组合。各单因素消融共享同一
v7 基线，v8 是将各组验证集选择合并后的配置；主程序的最终训练用于报告该联合组合的指标。

### Windows NVIDIA GPU 加速

训练和全部消融实验会自动选择设备。在 Windows 上安装 NVIDIA 驱动和 CUDA 版
PyTorch 后，无需修改代码；日志开头应显示类似：

```text
V1 device=cuda, gpu=NVIDIA ..., CUDA=..., AMP=True, TF32=True
```

这表示网络训练、监督预训练、Replay batch 和评估已经使用 GPU，其中 AMP 使用
FP16 混合精度，支持的显卡同时启用 TF32。若日志显示 `device=cpu`，说明当前
PyTorch 没有检测到 CUDA，代码会安全回退 CPU。单张 GPU 上各消融方案仍按顺序
运行；同时启动多个训练进程通常会争抢显存，并不会更快。设备和精度信息会写入
每次实验的 `resolved_config.json`。

从仓库根目录运行。完整训练默认 50 个 RL epoch，并且会跑满 50 轮：

```bash
python3 syj/lhz/mymodel_v1/main.py --episodes 50
```

小规模 smoke test（会产生一次最终 test 评估）：

```bash
python3 syj/lhz/mymodel_v1/main.py --episodes 1 --pretrain-epochs 1 --output-dir syj/lhz/mymodel_v1/runs/smoke
```

只检查数据划分、训练词表和产物，不训练：

```bash
python3 syj/lhz/mymodel_v1/main.py --dry-run --output-dir syj/lhz/mymodel_v1/runs/dry_run
```

恢复 `last.pt` 中的网络、优化器、replay 和随机数状态，并将新的训练结果写入新目录：

```bash
python3 syj/lhz/mymodel_v1/main.py --resume path/to/last.pt --episodes 20 --output-dir syj/lhz/mymodel_v1/runs/resumed
```

恢复时必须使用同一数据文件、`--split-seed` 和词表设置；恢复后的 `--episodes` 是新一轮训练预算。推理使用验证集最优的 `best.pt`：

```bash
python3 syj/lhz/mymodel_v1/predict_nontrace.py 胸闷 胸痛 畏寒 --model-path path/to/best.pt
python3 syj/lhz/mymodel_v1/predict_trace.py 胸闷 胸痛 畏寒 --model-path path/to/best.pt
```

## 结果与指标

每个 run 包含 `splits/` 下三份 CSV 和 `split_manifest.json`、完整参数 `resolved_config.json`、`best.pt`、`last.pt`、验证/测试 JSON、逐标签 CSV 和 `history.json`。选择模型时只看 validation；论文中报告 `test_metrics.json` 的一次性最终值，并报告多个训练随机种子的均值和标准差。

除 sample/micro/macro F1 外，评估会输出 `supported_macro_f1`、漏选率、过选率、停止深度、zero-F1 标签数，以及按 train support 划分的 `rare_1_4`、`few_5_19`、`medium_20_99`、`head_100_plus` 指标。验证或测试集中没有出现的标签不应被用于宏平均的结论。

训练曲线直接读取结构化历史，而不是旧日志格式：

```bash
python3 syj/lhz/mymodel_v1/plot_training.py --run-dir path/to/run
```

新版训练图包含 6 个面板：训练奖励、TD loss、四类验证 F1、精确率/召回率/完全匹配、停止与误选诊断，以及训练集/验证集平均标签数。`zero_f1_count` 仍保留在 JSON/CSV 指标中，但不再绘入训练曲线。图中不再绘制最佳 epoch 的绿色虚线，但标题和终端仍会报告最佳 epoch。默认完整跑满轮次，因此不显示红线；只有手动设置 `--patience` 为正数并实际触发早停时，才用红色点线标出早停 epoch。无论是否早停，最终测试始终使用验证集指标对应的最佳 checkpoint，而不是最后一轮模型。

也可以在 VS Code 中直接打开 `plot_training.py`、`predict_nontrace.py` 或
`predict_trace.py` 后点击“运行 Python 文件”。未指定参数时，它们会自动选择
`runs` 下最新的有效训练结果；两个预测入口随后会在终端中等待输入症状。

## 消融

详见 [`ablations/README.md`](ablations/README.md)。每个实验族使用相同的症状组划分、训练预算和 validation 选模规则，且每 seed 结果与跨 seed 汇总都在对应批次目录中。六组清洗数据消融已完成，无需再跑；现在应运行主程序得到 v8 联合配置的最终模型。
