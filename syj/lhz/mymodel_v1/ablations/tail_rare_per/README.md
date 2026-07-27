# Replay 端稀有度加权消融

## 正式协议

- 清洗数据批次：`results/tail_rare_per/20260720_094515`
- seed 9、17、29，共6次训练，全部完成。
- 两组都使用标准 TD-error PER；只比较 `rare_per_scale=0/0.5`。
- “无稀有加权”不等于关闭 PER；本组没有消融 `use_per=0`。

## 验证集结果

| Replay 策略 | Sample F1 | Supported Macro-F1 | Exact Match | 漏选率 | Zero-F1 |
|---|---:|---:|---:|---:|---:|
| **标准 PER，无稀有加权** | **0.7826 +/- 0.0011** | 0.5442 +/- 0.0240 | **0.2629** | 0.1907 | 3.00 |
| PER + 稀有加权 0.5 | 0.7798 +/- 0.0026 | **0.5446 +/- 0.0046** | 0.2585 | **0.1903** | **2.33** |

稀有加权的 Supported Macro-F1 仅高 0.0004，远小于跨 seed 波动，不能视为稳定改善。
它同时使 Sample F1 和 Exact Match 降低。两组的 Rare 1--4 Macro-F1 在三个 seed
上均为0，因此没有证据表明 0.5 加权改善了极稀有标签。

## 最终选择

```python
use_per = 1
rare_per_scale = 0.0
```

保留标准 TD-error PER，取消额外的稀有标签 priority multiplier。这只说明当前
`rare_per_scale=0.5` 无稳定收益，不代表所有 rarity-aware replay 方法都无效。

## 测试集报告

无稀有加权方案的 Sample F1 为 0.7991 +/- 0.0011，Supported Macro-F1 为
0.4063 +/- 0.0212；两项都高于加权方案。测试集不参与选型。

## 操作

需要复现时，直接点击本目录 `run.py`。
