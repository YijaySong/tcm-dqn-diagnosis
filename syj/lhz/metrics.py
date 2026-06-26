# -*- coding: utf-8 -*-
"""集合指标计算文件。

实现证候要素集合之间的TP/FP/FN、Precision、Recall、F1和Jaccard计算，
同时供环境奖励函数和最终评估指标使用。
"""


def calc_tp_fp_fn(selected, true):
    selected_set = set(selected)
    true_set = set(true)
    tp = len(selected_set & true_set)
    fp = len(selected_set - true_set)
    fn = len(true_set - selected_set)
    return tp, fp, fn


def set_precision(selected, true):
    tp, fp, _ = calc_tp_fp_fn(selected, true)
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def set_recall(selected, true):
    tp, _, fn = calc_tp_fp_fn(selected, true)
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def set_f1(selected, true):
    p = set_precision(selected, true)
    r = set_recall(selected, true)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def set_jaccard(selected, true):
    selected_set = set(selected)
    true_set = set(true)
    union = selected_set | true_set
    return len(selected_set & true_set) / len(union) if union else 0.0
