# LHZ 对比实验

本目录保存 LHZ 任务的对比模型。LHZ 是多标签证候要素集合预测任务，不是单标签多分类任务，因此所有模型都输出证候要素集合，并统一评估 `auto` 和 `top2` 两种模式。

## 模型列表

| 脚本 | 名称 | 类型 | 说明 |
|---|---|---|---|
| `run_frequency_prior.py` | Frequency Prior | 基线 | 按训练集标签频率固定排序预测 |
| `run_mlp_bce.py` | MLP-BCE | 神经网络基线 | 症状multi-hot输入 + BCEWithLogitsLoss |
| `run_logreg_ovr.py` | LogReg OvR | 经典机器学习 | One-vs-Rest Logistic Regression |
| `run_bernoulli_nb.py` | Bernoulli NB | 经典机器学习 | 适合二值症状特征的朴素贝叶斯 |
| `run_linear_svm.py` | Linear SVM OvR | 经典机器学习 | One-vs-Rest Linear SVM |
| `run_random_forest.py` | Random Forest OvR | 集成学习 | One-vs-Rest Random Forest |
| `run_hist_gradient_boosting.py` | HistGradientBoosting OvR | 较新机器学习 | sklearn HistGradientBoostingClassifier |

其中 sklearn 相关模型在未安装 scikit-learn 或版本不支持时会记录为 `skipped`，不会影响其他模型运行。

## auto 和 top2 含义

- `auto`：对非 RL 模型表示阈值式多标签输出。概率模型默认阈值为 0.5；Linear SVM 默认 margin 阈值为 0.0；如果没有标签超过阈值，则回退输出最高分标签。
- `top2`：固定输出得分最高的 2 个证候要素，用于和 DQN Top-2 评估对齐。

## 一键运行

```bash
python3 syj/lhz/contrast/run_all.py
```

结果保存到：

```text
syj/lhz/contrast/results/<时间戳>/
├── all_metrics.json
├── aggregate_metrics.csv
├── label_metrics.csv
├── contrast_metrics.pdf
├── frequency_prior/
├── mlp_bce/
├── logreg_ovr/
├── bernoulli_nb/
├── linear_svm/
├── random_forest/
└── hist_gradient_boosting/
```

常用命令：

```bash
# 只查看将要执行什么，不训练
python3 syj/lhz/contrast/run_all.py --dry-run

# 快速烟测，只运行频率基线和MLP，且MLP只训练1轮
python3 syj/lhz/contrast/run_all.py --models frequency_prior mlp_bce --run-id contrast_smoke --mlp-epochs 1 --no-plot

# 只运行经典模型中的两个
python3 syj/lhz/contrast/run_all.py --models logreg_ovr linear_svm

# 某个模型失败后继续跑后续模型
python3 syj/lhz/contrast/run_all.py --continue-on-error
```

## 单独运行某个模型

```bash
python3 syj/lhz/contrast/run_frequency_prior.py
python3 syj/lhz/contrast/run_mlp_bce.py --epochs 10
python3 syj/lhz/contrast/run_logreg_ovr.py
python3 syj/lhz/contrast/run_bernoulli_nb.py
python3 syj/lhz/contrast/run_linear_svm.py
python3 syj/lhz/contrast/run_random_forest.py
python3 syj/lhz/contrast/run_hist_gradient_boosting.py
```

## 可视化

`run_all.py` 默认会尝试生成 `contrast_metrics.pdf`。也可以手动运行：

```bash
python3 syj/lhz/contrast/plot_contrast_metrics.py
```

指定结果目录：

```bash
python3 syj/lhz/contrast/plot_contrast_metrics.py --result-dir syj/lhz/contrast/results/<时间戳>
```

同时输出 PNG：

```bash
python3 syj/lhz/contrast/plot_contrast_metrics.py --also-png
```

核心指标用图展示：

- `sample_f1`
- `sample_recall`
- `sample_precision`
- `exact_match`
- `micro_f1`
- `macro_f1`

辅助指标用表格展示：

- `sample_jaccard`
- `label_accuracy`
- `hamming_loss`
- `hit_rate`
- `empty_prediction_rate`
- `avg_selected_count`
- `avg_true_count`
- `avg_cardinality_error`
- `avg_over_select`
- `avg_under_select`

## 注意事项

- 所有模型复用 `syj/lhz/data.py` 的数据加载和划分逻辑。
- 所有模型复用 `syj/lhz/evaluation.py` 的多标签评估指标。
- 不使用 CrossEntropyLoss 或 softmax，因为本任务不是单标签多分类。
- 生成的 `results/` 目录已在本目录 `.gitignore` 中忽略。
