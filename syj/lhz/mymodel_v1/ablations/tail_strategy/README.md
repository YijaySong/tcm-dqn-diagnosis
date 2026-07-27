# 长尾策略说明

`tail_strategy` 不是第三组“组合消融”，也不测试监督损失 × rarity-PER 的全因子交互。它只是一个便捷入口：点击 `run.py` 后依次重复运行 `tail_supervised_loss` 和 `tail_rare_per`。

为避免混杂，长尾问题被拆成两个独立实验族：

- `tail_supervised_loss`：比较 plain BCE、class-balanced BCE、focal BCE，并固定 `rare_per_scale=0`。
- `tail_rare_per`：比较 `rare_per_scale=0/0.5`，并继承批次冻结的监督损失。

因此，只要两个子实验已经得到可用结果，就不需要再运行本目录 `run.py`；运行它不会产生新的组合证据，只会重新执行两批已有实验。

清洗数据上的最终结论是类别平衡 BCE + 标准 PER + `rare_per_scale=0`。
两个子实验都已完成，不建议再运行本目录，以免重复长时间训练。
