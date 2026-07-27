# 监督、示范与 RL 组成消融

## 正式协议

- 清洗数据批次：`results/supervision_components/20260719_101633`
- seed 9、17、29，7个方案共21次训练，全部完成。
- Windows 上的中文绘图问题不影响 CSV、JSON、checkpoint 或本次数值分析。
- 四个独立开关是监督预训练、标签示范 Replay warmup、辅助监督和 RL 更新。

## 方案定义

| 方案 | 预训练 | 示范 Replay | 辅助 BCE | RL |
|---|---:|---:|---:|---:|
| full_hybrid | 开 | 开 | 开，0.05 | 开 |
| no_pretrain | 关 | 开 | 开，0.05 | 开 |
| no_expert_warmup | 开 | 关 | 开，0.05 | 开 |
| no_aux_bce | 开 | 开 | 关 | 开 |
| pretrain_rl | 开 | 关 | 关 | 开 |
| supervised_only | 开 | 关 | 关 | 关 |
| pure_rl | 关 | 关 | 关 | 开 |

## 验证集结果

| 方案 | Sample F1 | Supported Macro-F1 | Exact Match | 漏选率 | Zero-F1 |
|---|---:|---:|---:|---:|---:|
| **full_hybrid** | **0.7806 +/- 0.0064** | 0.5567 +/- 0.0111 | **0.2576** | 0.2029 | 2.33 |
| no_pretrain | 0.7786 +/- 0.0044 | 0.5454 +/- 0.0126 | 0.2540 | **0.1971** | 2.67 |
| no_expert_warmup | 0.7762 +/- 0.0023 | 0.5432 +/- 0.0169 | 0.2406 | 0.2055 | 2.67 |
| no_aux_bce | 0.7595 +/- 0.0097 | **0.5622 +/- 0.0084** | 0.2228 | 0.2144 | **1.67** |
| pretrain_rl | 0.7579 +/- 0.0055 | 0.4665 +/- 0.0391 | 0.2335 | 0.2228 | 4.33 |
| supervised_only | 0.6499 +/- 0.0264 | 0.3097 +/- 0.0100 | 0.1266 | 0.4150 | 7.33 |
| pure_rl | 0.7054 +/- 0.0698 | 0.2578 +/- 0.0214 | 0.1622 | 0.3033 | 11.33 |

## 组件贡献

- **RL：必须保留。** `pretrain_rl` 相对 `supervised_only` 将 Sample F1 从 0.6499
  提高到 0.7579，Supported Macro-F1 从 0.3097 提高到 0.4665，漏选率降低 0.1923。
- **示范 Replay：保留。** `full_hybrid` 相对 `no_expert_warmup` 的 Sample F1、
  Supported Macro-F1 和 Exact Match 分别提高 0.0044、0.0135 和 0.0169。
- **监督预训练：保留。** 相对 `no_pretrain`，完整模型的 Supported Macro-F1
  提高 0.0113，Sample F1 也略高；虽然漏选率略差 0.0058，不足以抵消总体收益。
- **辅助监督：保留。** `no_aux_bce` 的 Supported Macro-F1 高 0.0055，但该差异
  小于跨 seed 波动；完整模型的 Sample F1 高 0.0212、Exact Match 高 0.0348，
  且漏选率更低。因此不机械地按单一均值去掉辅助监督。

## 最终选择

```python
use_pretrain = 1
pretrain_epochs = 10
use_expert_warmup = 1
use_aux_bce = 1
aux_bce_weight = 0.05
run_rl = 1
```

最终采用 `full_hybrid`。该方法是示范和辅助监督增强的监督--强化学习
混合模型，不是纯 RL 或无监督模型。

## 测试集报告

`full_hybrid` 的 Sample F1 为 0.7846 +/- 0.0036，Supported Macro-F1 为
0.4144 +/- 0.0104。测试集数值不参与方案选择。

## 操作

需要复现时，直接点击本目录 `run.py`。
