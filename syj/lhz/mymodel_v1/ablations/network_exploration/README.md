# 网络与探索策略消融

## 正式协议

- 清洗数据批次：`results/network_exploration/20260719_142310`
- seed 9、17、29，共9次训练，全部完成。
- 奖励、STOP、监督信号、数据划分和训练预算相同，只改变探索方式及对应网络层。
- Windows 或 macOS 的绘图字体警告不影响 JSON、CSV 和 checkpoint；本次数值来自结构化结果。

## 验证集结果

| 方案 | Sample F1 | Supported Macro-F1 | Exact Match | 漏选率 |
|---|---:|---:|---:|---:|
| Epsilon-greedy | 0.7694 +/- 0.0028 | 0.5305 +/- 0.0161 | 0.2335 | 0.2137 |
| NoisyNet | 0.7816 +/- 0.0049 | 0.5396 +/- 0.0100 | 0.2513 | **0.1900** |
| **无显式探索** | **0.7826 +/- 0.0011** | **0.5442 +/- 0.0240** | **0.2629** | 0.1907 |

无显式探索的 Sample F1、Supported Macro-F1 和 Exact Match 均值最高。
它与 NoisyNet 的宏平均误差范围重叠，因此不声称无探索显著更优；
但在没有稳定收益时，没有必要增加 NoisyLinear 或随机动作。

## 最终选择

```python
model_type = 'set_dueling'
exploration_mode = 'none'
eps_start = 0.0
eps_end = 0.0
```

主模型保留症状分支、已选标签分支、逐元交互、联合编码和 Dueling Value/Advantage
分支。NoisyNet 实现仅用于复现消融，主模型不启用。本组没有比较普通 Q
头与 Dueling Q 头，因此不用该结果单独声称 Dueling 优于所有其他结构。

## 测试集报告

最终方案的 Sample F1 为 0.7991 +/- 0.0011，Supported Macro-F1 为
0.4063 +/- 0.0212。NoisyNet 在测试集宏平均更高，但测试集不用于推翻验证集选型。

## 操作

需要复现时，直接点击本目录 `run.py`。
