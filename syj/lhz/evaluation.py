# -*- coding: utf-8 -*-
"""模型评估文件。

负责在测试集上调用训练好的DQN策略进行自主停止预测和固定Top-2预测，
并计算样本级、标签级、多标签整体Precision/Recall/F1等评估指标。
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


def evaluate_prediction_set(env, pred_action_sets, true_name_sets, title, logger):
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

    if precision_recall_fscore_support is not None and hamming_loss is not None:
        micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        label_p, label_r, label_f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        hamming = hamming_loss(y_true, y_pred)
    else:
        label_tp = (y_true * y_pred).sum(axis=0)
        label_fp = ((1 - y_true) * y_pred).sum(axis=0)
        label_fn = (y_true * (1 - y_pred)).sum(axis=0)
        label_p = np.divide(label_tp, label_tp + label_fp, out=np.zeros_like(label_tp, dtype=float), where=(label_tp + label_fp) != 0)
        label_r = np.divide(label_tp, label_tp + label_fn, out=np.zeros_like(label_tp, dtype=float), where=(label_tp + label_fn) != 0)
        label_f = np.divide(2 * label_p * label_r, label_p + label_r, out=np.zeros_like(label_p, dtype=float), where=(label_p + label_r) != 0)
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

    logger.info(f"**{title}")
    logger.info(f"**样本Jaccard avg:{np.mean(sample_j):.4f} max:{np.max(sample_j):.4f} min:{np.min(sample_j):.4f}")
    logger.info(f"**样本Precision avg:{np.mean(sample_p):.4f}, Recall avg:{np.mean(sample_r):.4f}, F1 avg:{np.mean(sample_f):.4f}")
    logger.info(f"**ExactMatch:{np.mean(exact_match):.4f}, HammingLoss:{hamming:.4f}")
    logger.info(f"**Micro P/R/F1:{micro_p:.4f}/{micro_r:.4f}/{micro_f:.4f}")
    logger.info(f"**Macro P/R/F1:{macro_p:.4f}/{macro_r:.4f}/{macro_f:.4f}")
    logger.info(f"**平均推荐数:{np.mean(selected_count):.4f}, 平均真实数:{np.mean(true_count):.4f}, 平均数量误差:{np.mean(cardinality_error):.4f}")
    for action_idx in range(env.Se_action_num):
        logger.info(f"**标签[{env.action_space[action_idx]}] P/R/F1:{label_p[action_idx]:.4f}/{label_r[action_idx]:.4f}/{label_f[action_idx]:.4f}")
    logger.info("")

    return {
        'sample_jaccard': float(np.mean(sample_j)),
        'sample_precision': float(np.mean(sample_p)),
        'sample_recall': float(np.mean(sample_r)),
        'sample_f1': float(np.mean(sample_f)),
        'exact_match': float(np.mean(exact_match)),
        'hamming_loss': float(hamming),
        'micro_f1': float(micro_f),
        'macro_f1': float(macro_f),
        'avg_selected_count': float(np.mean(selected_count)),
        'avg_cardinality_error': float(np.mean(cardinality_error)),
        'n_test': n_test,
    }


def evaluate(env, policy_net, action_selector, device, eval_data, logger, dataset_name="测试集"):
    logger.info(f"========== {dataset_name}评估开始 ==========")
    if len(eval_data) == 0:
        logger.info(f"{dataset_name}数据为空，跳过评估")
        empty_metrics = {'sample_f1': 0.0, 'exact_match': 0.0, 'micro_f1': 0.0, 'macro_f1': 0.0}
        return {'auto': empty_metrics, 'top2': empty_metrics}

    auto_pred_actions = []
    top2_pred_actions = []
    true_name_sets = []

    for data_piece in eval_data:
        state_np = env.reset(data_piece).copy()
        auto_pred_actions.append(
            predict_actions_from_state(env, policy_net, action_selector, device, state_np, force_top_k=None)
        )

        state_np = env.reset(data_piece).copy()
        top2_pred_actions.append(
            predict_actions_from_state(
                env, policy_net, action_selector, device, state_np,
                force_top_k=min(2, env.Se_action_num), max_actions=2
            )
        )

        true_name_sets.append(data_piece[1])

    auto_metrics = evaluate_prediction_set(env, auto_pred_actions, true_name_sets, f"{dataset_name}-模型自主停止", logger)
    top2_metrics = evaluate_prediction_set(env, top2_pred_actions, true_name_sets, f"{dataset_name}-固定Top-2诊断", logger)
    logger.info(f"**{dataset_name} 自主停止 vs Top-2 样本F1: {auto_metrics['sample_f1']:.4f} / {top2_metrics['sample_f1']:.4f}")
    logger.info("========== {0}评估结束 ==========".format(dataset_name))
    return {'auto': auto_metrics, 'top2': top2_metrics}
