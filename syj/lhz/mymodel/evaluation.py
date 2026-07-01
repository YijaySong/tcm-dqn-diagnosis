# -*- coding: utf-8 -*-
"""模型评估文件。

负责在测试集上调用训练好的DQN策略进行自主停止预测和固定Top-2预测，
并计算样本级、标签级、多标签整体Precision/Recall/F1等评估指标。

最重要的评估指标优先看这几个：
1. 样本F1（sample_f1）：每个病例先算Precision/Recall/F1再平均，
   综合衡量错选和漏选，是本任务最直观的核心指标，越高越好。
2. 样本Recall（sample_recall）：每个病例真实证候要素有多少被找回，
   重点反映漏诊/漏选情况，越高越好。
3. 样本Precision（sample_precision）：每个病例推荐出的证候要素有多少是真的，
   重点反映误诊/错选情况，越高越好。
4. ExactMatch / 严格准确率（exact_match）：预测集合与真实集合完全一致的病例比例，
   是最严格的准确率，越高越好，但多标签任务中通常会偏低。
5. Macro F1（macro_f1）：先算每个证候要素标签的F1再平均，更能反映低频标签表现，
   越高越好。

辅助诊断指标：
6. 样本Jaccard（sample_jaccard）：预测集合与真实集合的交并比，衡量集合重合程度，
   越高越好；它和sample_f1含义接近，因此主要作为辅助参考。
7. 标签级Accuracy（label_accuracy）：所有“病例-标签”二分类位置中预测正确的比例，
   越高越好；标签稀疏时容易被大量真阴性抬高，所以不能单独作为核心指标。
8. HammingLoss（hamming_loss）：所有“病例-标签”位置中预测错误的比例，
   是标签级错误率，越低越好。
9. Micro Precision / Recall / F1（micro_precision, micro_recall, micro_f1）：
   把所有标签的TP/FP/FN汇总后计算，更受高频标签影响，越高越好。
10. HitRate（hit_rate）：每个病例至少命中一个真实证候要素的比例，
    反映模型是否有基本命中能力，越高越好。
11. EmptyPredictionRate（empty_prediction_rate）：模型没有推荐任何证候要素的病例比例，
    通常越低越好。
12. 平均推荐数 / 平均真实数（avg_selected_count, avg_true_count）：
    用于观察模型整体推荐数量是否偏多或偏少，本身无绝对好坏。
13. 平均数量误差 / 平均多选数 / 平均漏选数（avg_cardinality_error,
    avg_over_select, avg_under_select）：反映预测数量偏差，越低越好。
14. 标签级P/R/F1/Support/PredCount（日志中逐标签输出；数组变量: label_p, label_r,
    label_f, label_support, label_pred_count）：定位每个证候要素预测好坏；P/R/F1越高越好。
"""

import importlib

import numpy as np

from metrics import set_jaccard, set_precision, set_recall
from state import Se_names_to_multihot, actions_to_multihot, predict_actions_from_state

try:
    sklearn_metrics = importlib.import_module('sklearn.metrics')
    precision_recall_fscore_support = sklearn_metrics.precision_recall_fscore_support
    hamming_loss = sklearn_metrics.hamming_loss
except ImportError:
    precision_recall_fscore_support = None
    hamming_loss = None


def eval_jaccard(selected, true):
    return set_jaccard(selected, true)


def eval_precision(selected, true):
    return set_precision(selected, true)


def eval_recall(selected, true):
    return set_recall(selected, true)


def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0
    )


def evaluate_prediction_set(env, pred_action_sets, true_name_sets, title, logger, trace_sets=None):
    n_test = len(true_name_sets)
    y_pred = np.array([actions_to_multihot(env, actions) for actions in pred_action_sets])
    y_true = np.array([Se_names_to_multihot(env, names) for names in true_name_sets])

    sample_j = []
    sample_p = []
    sample_r = []
    sample_f = []
    exact_match = []
    selected_count = []
    true_count = []
    cardinality_error = []
    over_select_count = []
    under_select_count = []
    hit_match = []
    empty_prediction = []

    for pred_actions, true_names in zip(pred_action_sets, true_name_sets):
        pred_names = [env.action_space[action] for action in pred_actions]
        p = eval_precision(pred_names, true_names)
        r = eval_recall(pred_names, true_names)
        f = eval_f1(p, r)
        sample_j.append(eval_jaccard(pred_names, true_names))
        sample_p.append(p)
        sample_r.append(r)
        sample_f.append(f)
        exact_match.append(1 if set(pred_names) == set(true_names) else 0)
        selected_count.append(len(pred_names))
        true_count.append(len(true_names))
        cardinality_error.append(abs(len(pred_names) - len(true_names)))
        over_select_count.append(max(0, len(pred_names) - len(true_names)))
        under_select_count.append(max(0, len(true_names) - len(pred_names)))
        hit_match.append(1 if set(pred_names) & set(true_names) else 0)
        empty_prediction.append(1 if len(pred_names) == 0 else 0)

    label_tp = (y_true * y_pred).sum(axis=0)
    label_fp = ((1 - y_true) * y_pred).sum(axis=0)
    label_fn = (y_true * (1 - y_pred)).sum(axis=0)
    label_tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
    label_support = y_true.sum(axis=0)
    label_pred_count = y_pred.sum(axis=0)
    label_accuracy = safe_divide(label_tp + label_tn, label_tp + label_fp + label_fn + label_tn)

    if precision_recall_fscore_support is not None and hamming_loss is not None:
        micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        label_p, label_r, label_f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        hamming = hamming_loss(y_true, y_pred)
    else:
        label_p = safe_divide(label_tp, label_tp + label_fp)
        label_r = safe_divide(label_tp, label_tp + label_fn)
        label_f = safe_divide(2 * label_p * label_r, label_p + label_r)
        total_tp = label_tp.sum()
        total_fp = label_fp.sum()
        total_fn = label_fn.sum()
        micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0.0
        macro_p = float(np.mean(label_p))
        macro_r = float(np.mean(label_r))
        macro_f = float(np.mean(label_f))
        hamming = float(np.not_equal(y_true, y_pred).mean())

    exact_match_avg = float(np.mean(exact_match))
    label_accuracy_micro = float((label_tp.sum() + label_tn.sum()) / y_true.size) if y_true.size else 0.0
    hit_rate = float(np.mean(hit_match))
    empty_prediction_rate = float(np.mean(empty_prediction))
    avg_over_select = float(np.mean(over_select_count))
    avg_under_select = float(np.mean(under_select_count))

    stop_depths = []
    stop_margins = []
    premature_stop = []
    late_stop = []
    if trace_sets is not None:
        for pred_actions, true_names, trace in zip(pred_action_sets, true_name_sets, trace_sets):
            stop_step = len(trace) if trace and trace[-1].get('is_stop') else len(pred_actions)
            stop_depths.append(stop_step)
            if trace:
                stop_margins.append(trace[-1].get('stop_margin', 0.0))
            premature_stop.append(1 if len(pred_actions) < len(true_names) else 0)
            late_stop.append(1 if len(pred_actions) > len(true_names) else 0)
    avg_stop_depth = float(np.mean(stop_depths)) if stop_depths else 0.0
    avg_stop_margin = float(np.mean(stop_margins)) if stop_margins else 0.0
    premature_stop_rate = float(np.mean(premature_stop)) if premature_stop else 0.0
    late_stop_rate = float(np.mean(late_stop)) if late_stop else 0.0

    logger.info(f"**{title}")
    logger.info(f"**核心指标 sample_f1:{np.mean(sample_f):.4f}, sample_recall:{np.mean(sample_r):.4f}, sample_precision:{np.mean(sample_p):.4f}, exact_match:{exact_match_avg:.4f}, macro_f1:{macro_f:.4f}")
    logger.info(f"**样本Jaccard(sample_jaccard) avg:{np.mean(sample_j):.4f} max:{np.max(sample_j):.4f} min:{np.min(sample_j):.4f}")
    logger.info(f"**样本Precision avg:{np.mean(sample_p):.4f}, Recall avg:{np.mean(sample_r):.4f}, F1 avg:{np.mean(sample_f):.4f}")
    logger.info(f"**ExactMatch/严格准确率(exact_match):{exact_match_avg:.4f}, 标签级Accuracy(label_accuracy):{label_accuracy_micro:.4f}, HammingLoss(hamming_loss):{hamming:.4f}")
    logger.info(f"**Micro P/R/F1:{micro_p:.4f}/{micro_r:.4f}/{micro_f:.4f}")
    logger.info(f"**Macro P/R/F1:{macro_p:.4f}/{macro_r:.4f}/{macro_f:.4f}")
    logger.info(f"**HitRate(hit_rate):{hit_rate:.4f}, EmptyPredictionRate(empty_prediction_rate):{empty_prediction_rate:.4f}")
    logger.info(f"**平均推荐数:{np.mean(selected_count):.4f}, 平均真实数:{np.mean(true_count):.4f}, 平均数量误差:{np.mean(cardinality_error):.4f}, 平均多选数:{avg_over_select:.4f}, 平均漏选数:{avg_under_select:.4f}")
    if trace_sets is not None:
        logger.info(f"**停止诊断 avg_stop_depth:{avg_stop_depth:.4f}, avg_stop_margin:{avg_stop_margin:.4f}, premature_stop_rate:{premature_stop_rate:.4f}, late_stop_rate:{late_stop_rate:.4f}")
    for action_idx in range(env.Se_action_num):
        logger.info(
            f"**标签[{env.action_space[action_idx]}] "
            f"P/R/F1/Acc/Support/PredCount:"
            f"{label_p[action_idx]:.4f}/{label_r[action_idx]:.4f}/{label_f[action_idx]:.4f}/"
            f"{label_accuracy[action_idx]:.4f}/"
            f"{int(label_support[action_idx])}/{int(label_pred_count[action_idx])}"
        )
    logger.info("")

    return {
        'sample_jaccard': float(np.mean(sample_j)),
        'sample_precision': float(np.mean(sample_p)),
        'sample_recall': float(np.mean(sample_r)),
        'sample_f1': float(np.mean(sample_f)),
        'exact_match': exact_match_avg,
        'label_accuracy': label_accuracy_micro,
        'hamming_loss': float(hamming),
        'micro_precision': float(micro_p),
        'micro_recall': float(micro_r),
        'micro_f1': float(micro_f),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f),
        'hit_rate': hit_rate,
        'empty_prediction_rate': empty_prediction_rate,
        'avg_selected_count': float(np.mean(selected_count)),
        'avg_true_count': float(np.mean(true_count)),
        'avg_cardinality_error': float(np.mean(cardinality_error)),
        'avg_over_select': avg_over_select,
        'avg_under_select': avg_under_select,
        'avg_stop_depth': avg_stop_depth,
        'avg_stop_margin': avg_stop_margin,
        'premature_stop_rate': premature_stop_rate,
        'late_stop_rate': late_stop_rate,
        'n_test': n_test,
    }


def evaluate(
    env, policy_net, action_selector, device, eval_data, logger, dataset_name="测试集",
    stop_margin_threshold=None, min_actions=0,
):
    logger.info(f"========== {dataset_name}评估开始 ==========")
    if len(eval_data) == 0:
        logger.info(f"{dataset_name}数据为空，跳过评估")
        empty_metrics = {'sample_f1': 0.0, 'exact_match': 0.0, 'micro_f1': 0.0, 'macro_f1': 0.0}
        return {'auto': empty_metrics, 'top2': empty_metrics}

    auto_pred_actions = []
    auto_traces = []
    top2_pred_actions = []
    top2_traces = []
    true_name_sets = []

    for data_piece in eval_data:
        state_np = env.reset(data_piece).copy()
        auto_actions, auto_trace = predict_actions_from_state(
            env, policy_net, action_selector, device, state_np, force_top_k=None,
            stop_margin_threshold=stop_margin_threshold, min_actions=min_actions,
            return_trace=True,
        )
        auto_pred_actions.append(auto_actions)
        auto_traces.append(auto_trace)

        state_np = env.reset(data_piece).copy()
        top2_actions, top2_trace = predict_actions_from_state(
            env, policy_net, action_selector, device, state_np,
            force_top_k=min(2, env.Se_action_num), max_actions=2,
            return_trace=True,
        )
        top2_pred_actions.append(top2_actions)
        top2_traces.append(top2_trace)

        true_name_sets.append(data_piece[1])

    auto_metrics = evaluate_prediction_set(env, auto_pred_actions, true_name_sets, f"{dataset_name}-模型自主停止", logger, trace_sets=auto_traces)
    top2_metrics = evaluate_prediction_set(env, top2_pred_actions, true_name_sets, f"{dataset_name}-固定Top-2诊断", logger, trace_sets=top2_traces)
    logger.info(f"**{dataset_name} 自主停止 vs Top-2 样本F1: {auto_metrics['sample_f1']:.4f} / {top2_metrics['sample_f1']:.4f}")
    logger.info("========== {0}评估结束 ==========".format(dataset_name))
    return {'auto': auto_metrics, 'top2': top2_metrics}
