# -*- coding: utf-8 -*-
"""V1 结构化评估：总体、逐标签、训练支持度频段、zero-F1 和 STOP 诊断。"""

import numpy as np

from metrics import set_jaccard, set_precision, set_recall
from state import Se_names_to_multihot, actions_to_multihot, predict_actions_from_state


def _safe_divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(np.asarray(numerator, dtype=float)), where=np.asarray(denominator) != 0)


def _band_for_support(support):
    if support <= 4:
        return 'rare_1_4'
    if support <= 19:
        return 'few_5_19'
    if support <= 99:
        return 'medium_20_99'
    return 'head_100_plus'


def evaluate_prediction_set(env, predictions, truths, traces, logger, title):
    y_pred = np.asarray([actions_to_multihot(env, actions) for actions in predictions])
    y_true = np.asarray([Se_names_to_multihot(env, labels) for labels in truths])
    sample_p, sample_r, sample_f, jaccards, exact = [], [], [], [], []
    selected_counts, true_counts, over, under, hit, empty = [], [], [], [], [], []
    for actions, labels in zip(predictions, truths):
        predicted_names = [env.action_space[action] for action in actions]
        precision = set_precision(predicted_names, labels)
        recall = set_recall(predicted_names, labels)
        sample_p.append(precision)
        sample_r.append(recall)
        sample_f.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        jaccards.append(set_jaccard(predicted_names, labels))
        exact.append(float(set(predicted_names) == set(labels)))
        selected_counts.append(len(predicted_names))
        true_counts.append(len(labels))
        over.append(max(0, len(predicted_names) - len(labels)))
        under.append(max(0, len(labels) - len(predicted_names)))
        hit.append(float(bool(set(predicted_names) & set(labels))))
        empty.append(float(not predicted_names))

    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
    label_p = _safe_divide(tp, tp + fp)
    label_r = _safe_divide(tp, tp + fn)
    label_f = _safe_divide(2 * label_p * label_r, label_p + label_r)
    eval_support = y_true.sum(axis=0)
    pred_count = y_pred.sum(axis=0)
    supported = eval_support > 0
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    label_metrics, band_rows = [], {}
    for action in range(env.Se_action_num):
        train_support = int(env.Se_supports.get(action, 0))
        row = {
            'label': env.action_space[action], 'train_support': train_support,
            'eval_support': int(eval_support[action]), 'pred_count': int(pred_count[action]),
            'tp': int(tp[action]), 'fp': int(fp[action]), 'fn': int(fn[action]),
            'precision': float(label_p[action]), 'recall': float(label_r[action]), 'f1': float(label_f[action]),
            'frequency_band': _band_for_support(train_support),
        }
        label_metrics.append(row)
    for band in ('rare_1_4', 'few_5_19', 'medium_20_99', 'head_100_plus'):
        rows = [row for row in label_metrics if row['frequency_band'] == band]
        supported_rows = [row for row in rows if row['eval_support'] > 0]
        band_rows[band] = {
            'label_count': len(rows), 'eval_supported_label_count': len(supported_rows),
            'macro_f1': float(np.mean([row['f1'] for row in supported_rows])) if supported_rows else 0.0,
            'macro_recall': float(np.mean([row['recall'] for row in supported_rows])) if supported_rows else 0.0,
            'zero_f1_count': sum(row['f1'] == 0.0 for row in supported_rows),
            'never_predicted_count': sum(row['pred_count'] == 0 for row in supported_rows),
        }

    stop_depth = [len(trace) for trace in traces]
    stop_margin = [trace[-1].get('stop_margin', 0.0) if trace else 0.0 for trace in traces]
    metrics = {
        'sample_jaccard': float(np.mean(jaccards)), 'sample_precision': float(np.mean(sample_p)),
        'sample_recall': float(np.mean(sample_r)), 'sample_f1': float(np.mean(sample_f)),
        'exact_match': float(np.mean(exact)), 'micro_precision': float(micro_p),
        'micro_recall': float(micro_r), 'micro_f1': float(micro_f),
        'macro_precision': float(np.mean(label_p)), 'macro_recall': float(np.mean(label_r)),
        'macro_f1': float(np.mean(label_f)),
        'supported_macro_precision': float(np.mean(label_p[supported])) if supported.any() else 0.0,
        'supported_macro_recall': float(np.mean(label_r[supported])) if supported.any() else 0.0,
        'supported_macro_f1': float(np.mean(label_f[supported])) if supported.any() else 0.0,
        'supported_label_count': int(supported.sum()), 'zero_f1_count': sum(row['f1'] == 0.0 for row in label_metrics if row['eval_support'] > 0),
        'zero_f1_rate': float(np.mean([row['f1'] == 0.0 for row in label_metrics if row['eval_support'] > 0])) if supported.any() else 0.0,
        'hit_rate': float(np.mean(hit)), 'empty_prediction_rate': float(np.mean(empty)),
        'avg_selected_count': float(np.mean(selected_counts)), 'avg_true_count': float(np.mean(true_counts)),
        'avg_cardinality_error': float(np.mean(np.abs(np.asarray(selected_counts) - np.asarray(true_counts)))),
        'avg_over_select': float(np.mean(over)), 'avg_under_select': float(np.mean(under)),
        'false_selection_rate': float(np.mean([1 - value for value in sample_p])),
        'miss_selection_rate': float(np.mean([1 - value for value in sample_r])),
        'avg_stop_depth': float(np.mean(stop_depth)), 'avg_stop_margin': float(np.mean(stop_margin)),
        'premature_stop_rate': float(np.mean(np.asarray(selected_counts) < np.asarray(true_counts))),
        'late_stop_rate': float(np.mean(np.asarray(selected_counts) > np.asarray(true_counts))),
        'n_samples': len(truths),
    }
    logger.info(f'**{title}: sample_f1={metrics["sample_f1"]:.4f}, exact_match={metrics["exact_match"]:.4f}, '
                f'micro_f1={metrics["micro_f1"]:.4f}, macro_f1={metrics["macro_f1"]:.4f}, '
                f'supported_macro_f1={metrics["supported_macro_f1"]:.4f}, zero_f1={metrics["zero_f1_count"]}')
    logger.info(f'**停止: depth={metrics["avg_stop_depth"]:.4f}, premature={metrics["premature_stop_rate"]:.4f}, '
                f'late={metrics["late_stop_rate"]:.4f}, 漏选率={metrics["miss_selection_rate"]:.4f}')
    return {'auto': metrics, 'label_metrics': label_metrics, 'band_metrics': band_rows}


def evaluate(env, policy_net, device, eval_data, logger, dataset_name='验证集',
             stop_margin_threshold=None, min_actions=1):
    if not eval_data:
        return {'auto': {'sample_f1': 0.0, 'supported_macro_f1': 0.0}, 'label_metrics': [], 'band_metrics': {}}
    predictions, traces, truths = [], [], []
    for symptoms, labels in eval_data:
        state = env.reset((symptoms, labels)).copy()
        actions, trace = predict_actions_from_state(env, policy_net, device, state,
                                                    stop_margin_threshold=stop_margin_threshold,
                                                    min_actions=min_actions, return_trace=True)
        predictions.append(actions)
        traces.append(trace)
        truths.append(labels)
    return evaluate_prediction_set(env, predictions, truths, traces, logger, f'{dataset_name}-自主停止')
