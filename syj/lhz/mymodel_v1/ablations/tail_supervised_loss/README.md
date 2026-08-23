# 监督损失消融

## 正式协议

- 清洗数据 Windows 批次：`results/tail_supervised_loss/20260720_094527`
- 普通 BCE 和类别平衡 BCE 各完成3个 seed；Focal BCE 完成 seed 9、29。
- Focal seed 17 在专家 Replay 预填充后进程异常退出，无验证或测试指标，已按缺失 seed 排除。
- 本批次的其余8次训练有效，不因一次失败作废，也不需要全部重跑。
- 三组固定 `rare_per_scale=0`，只改变监督损失。

## 验证集结果

| 损失 | 完成 seed | Sample F1 | Supported Macro-F1 | Exact Match | 漏选率 | Zero-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 普通 BCE | 3 | 0.7806 +/- 0.0050 | 0.5342 +/- 0.0111 | 0.2487 | **0.1925** | 2.67 |
| **类别平衡 BCE** | **3** | **0.7825 +/- 0.0061** | **0.5441 +/- 0.0133** | **0.2665** | 0.2032 | 2.67 |
| Focal BCE | 2 | 0.7763 +/- 0.0036 | 0.5403 +/- 0.0127 | 0.2527 | 0.2035 | **2.50** |

类别平衡 BCE 在完整的3个 seed 上同时取得最高的 Sample F1、Supported Macro-F1
和 Exact Match。Focal 缺失一个 seed，但现有2个 seed 的均值仍未超过类别平衡 BCE，
因此失败项不影响本次选型。

Rare 1--4 频段中，类别平衡 BCE 和 Focal 的均值都为0；普通 BCE 的
0.0667 仅来自 seed 9 的一次非零结果，另两个 seed 为0。该频段样本极少，
不能据此选择普通 BCE。

## 最终选择

```python
supervised_loss = 'class_balanced'
```

类别平衡 BCE 的 `pos_weight` 仅由训练集标签支持度计算并截断。
`focal_gamma=2.0` 保留为 Focal 备选实验参数，主模型不使用聚焦调制项。

## 测试集报告

类别平衡 BCE 的 Sample F1 为 0.7931 +/- 0.0041，Supported Macro-F1 为
0.4034 +/- 0.0089。测试集不参与损失选择。

## 操作

需要复现时，直接点击本目录 `run.py`。
