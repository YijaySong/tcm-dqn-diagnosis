# -*- coding: utf-8 -*-
"""LHZ对比实验通用工具。"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CONTRAST_DIR = Path(__file__).resolve().parent
LHZ_DIR = CONTRAST_DIR.parent
PROJECT_ROOT = LHZ_DIR.parent.parent
RESULTS_DIR = CONTRAST_DIR / 'results'

if str(LHZ_DIR) not in sys.path:
    sys.path.insert(0, str(LHZ_DIR))

from data import export_split_data, get_tcm_data, stratified_split_data, strip_sample_id  # noqa: E402
from env import Environment  # noqa: E402
from evaluation import evaluate_prediction_set, safe_divide  # noqa: E402
from state import Se_names_to_multihot, actions_to_multihot  # noqa: E402
from utils import default_data_path, set_seed  # noqa: E402

AGGREGATE_COLUMNS = [
    'experiment',
    'label',
    'mode',
    'sample_f1',
    'sample_recall',
    'sample_precision',
    'sample_jaccard',
    'exact_match',
    'label_accuracy',
    'hamming_loss',
    'micro_precision',
    'micro_recall',
    'micro_f1',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'hit_rate',
    'empty_prediction_rate',
    'avg_selected_count',
    'avg_true_count',
    'avg_cardinality_error',
    'avg_over_select',
    'avg_under_select',
]

LABEL_COLUMNS = [
    'experiment',
    'label',
    'mode',
    'syndrome_element',
    'precision',
    'recall',
    'f1',
    'accuracy',
    'support',
    'pred_count',
]

DEFAULT_SEED = 9
DEFAULT_TEST_RATIO = 0.2
DEFAULT_TOP_K = 2


def current_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_logger(log_path):
    logger_name = f'contrast_{Path(log_path).stem}_{id(log_path)}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s')

    file_handler = logging.FileHandler(log_path, mode='w+', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def model_output_dir(model_name, run_id=None, output_dir=None):
    if output_dir:
        return ensure_dir(output_dir)
    run_id = run_id or current_timestamp()
    return ensure_dir(RESULTS_DIR / run_id / model_name)


def add_common_arguments(parser):
    parser.add_argument('--run-id', default=None, help='结果批次目录名；默认使用当前时间戳')
    parser.add_argument('--output-dir', default=None, help='当前模型的输出目录；run_all.py会传入该参数')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--test-ratio', type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument('--max-se-num', type=int, default=0)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--threshold', type=float, default=0.5, help='auto模式阈值，概率模型默认0.5')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K, help='top-k固定输出数量，默认2')
    return parser


def prepare_dataset(seed, test_ratio, max_se_num, data_path, output_dir, logger):
    set_seed(seed)
    data_path = data_path or default_data_path()
    tcm_data, symptoms, Se, max_Se_len = get_tcm_data(data_path, max_Se_num=max_se_num, logger=logger)
    env = Environment(symptoms, Se)
    train_raw, test_raw = stratified_split_data(tcm_data, seed, test_ratio=test_ratio, logger=logger)
    export_split_data(train_raw, test_raw, output_dir / 'split_data', seed, test_ratio, logger=logger)
    train_data = strip_sample_id(train_raw)
    test_data = strip_sample_id(test_raw)
    logger.info(f'训练数据数量:{len(train_data)}, 测试数据数量:{len(test_data)}, 最大真实证候要素数:{max_Se_len}')
    return {
        'env': env,
        'train_data': train_data,
        'test_data': test_data,
        'raw_train_data': train_raw,
        'raw_test_data': test_raw,
        'symptoms': symptoms,
        'Se': Se,
        'max_Se_len': max_Se_len,
        'data_path': data_path,
    }


def vectorize_features(env, samples):
    x = np.zeros((len(samples), env.symp_len), dtype=np.float32)
    for row_idx, (symptoms, _) in enumerate(samples):
        for symptom in symptoms:
            state_idx = env.swapped_state_space.get(symptom)
            if state_idx is not None and state_idx < env.symp_len:
                x[row_idx, state_idx] = 1.0
    return x


def vectorize_labels(env, samples):
    y = np.zeros((len(samples), env.Se_action_num), dtype=np.float32)
    for row_idx, (_, Se_names) in enumerate(samples):
        for Se_name in Se_names:
            action_idx = env.swapped_action_space.get(Se_name)
            if action_idx is not None and action_idx < env.Se_action_num:
                y[row_idx, action_idx] = 1.0
    return y


def topk_indices(row, top_k):
    top_k = max(1, min(int(top_k), len(row)))
    return np.argsort(-row)[:top_k].astype(int).tolist()


def scores_to_actions(scores, mode, threshold=0.5, top_k=2, threshold_is_margin=False):
    scores = np.asarray(scores, dtype=float)
    predictions = []
    for row in scores:
        if mode == 'top2':
            predictions.append(topk_indices(row, top_k))
            continue
        if mode != 'auto':
            raise ValueError(f'未知预测模式: {mode}')
        selected = np.where(row >= threshold)[0].astype(int).tolist()
        if not selected:
            selected = [int(np.argmax(row))]
        predictions.append(selected)
    return predictions


def probability_scores_from_estimator(estimator, x_test):
    if hasattr(estimator, 'predict_proba'):
        prob = estimator.predict_proba(x_test)
        if isinstance(prob, list):
            cols = []
            for item in prob:
                arr = np.asarray(item)
                if arr.ndim == 2 and arr.shape[1] > 1:
                    cols.append(arr[:, 1])
                else:
                    cols.append(arr.reshape(-1))
            return np.stack(cols, axis=1).astype(float)
        return np.asarray(prob, dtype=float)
    if hasattr(estimator, 'decision_function'):
        return np.asarray(estimator.decision_function(x_test), dtype=float)
    return np.asarray(estimator.predict(x_test), dtype=float)


def compute_label_metrics(env, pred_action_sets, true_name_sets):
    y_pred = np.array([actions_to_multihot(env, actions) for actions in pred_action_sets])
    y_true = np.array([Se_names_to_multihot(env, names) for names in true_name_sets])
    label_tp = (y_true * y_pred).sum(axis=0)
    label_fp = ((1 - y_true) * y_pred).sum(axis=0)
    label_fn = (y_true * (1 - y_pred)).sum(axis=0)
    label_tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
    label_support = y_true.sum(axis=0)
    label_pred_count = y_pred.sum(axis=0)
    label_p = safe_divide(label_tp, label_tp + label_fp)
    label_r = safe_divide(label_tp, label_tp + label_fn)
    label_f = safe_divide(2 * label_p * label_r, label_p + label_r)
    label_accuracy = safe_divide(label_tp + label_tn, label_tp + label_fp + label_fn + label_tn)

    rows = []
    for action_idx in range(env.Se_action_num):
        rows.append({
            'syndrome_element': env.action_space[action_idx],
            'precision': float(label_p[action_idx]),
            'recall': float(label_r[action_idx]),
            'f1': float(label_f[action_idx]),
            'accuracy': float(label_accuracy[action_idx]),
            'support': int(label_support[action_idx]),
            'pred_count': int(label_pred_count[action_idx]),
        })
    return rows


def evaluate_contrast_predictions(env, test_data, auto_pred_actions, top2_pred_actions, logger):
    true_name_sets = [item[1] for item in test_data]
    logger.info('========== 测试集评估开始 ==========',)
    auto_metrics = evaluate_prediction_set(env, auto_pred_actions, true_name_sets, '测试集-模型自主停止', logger)
    top2_metrics = evaluate_prediction_set(env, top2_pred_actions, true_name_sets, '测试集-固定Top-2诊断', logger)
    logger.info(f"**测试集 自主停止 vs Top-2 样本F1: {auto_metrics['sample_f1']:.4f} / {top2_metrics['sample_f1']:.4f}")
    logger.info('========== 测试集评估结束 ==========')
    return {
        'auto': {
            'metrics': auto_metrics,
            'labels': compute_label_metrics(env, auto_pred_actions, true_name_sets),
        },
        'top2': {
            'metrics': top2_metrics,
            'labels': compute_label_metrics(env, top2_pred_actions, true_name_sets),
        },
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows, columns):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in columns})


def flatten_result_rows(result):
    aggregate_rows = []
    label_rows = []
    if result.get('status') != 'done':
        return aggregate_rows, label_rows
    experiment = result['experiment']
    label = result['label']
    for mode in ['auto', 'top2']:
        mode_data = result.get('metrics', {}).get(mode, {})
        metrics = mode_data.get('metrics', {})
        row = {'experiment': experiment, 'label': label, 'mode': mode}
        for column in AGGREGATE_COLUMNS:
            if column not in row:
                row[column] = metrics.get(column, '')
        aggregate_rows.append(row)
        for item in mode_data.get('labels', []):
            label_row = {'experiment': experiment, 'label': label, 'mode': mode}
            label_row.update(item)
            label_rows.append(label_row)
    return aggregate_rows, label_rows


def write_batch_csvs(batch_dir, results):
    aggregate_rows = []
    label_rows = []
    for result in results:
        aggregate, labels = flatten_result_rows(result)
        aggregate_rows.extend(aggregate)
        label_rows.extend(labels)
    write_csv(Path(batch_dir) / 'aggregate_metrics.csv', aggregate_rows, AGGREGATE_COLUMNS)
    write_csv(Path(batch_dir) / 'label_metrics.csv', label_rows, LABEL_COLUMNS)
    return aggregate_rows, label_rows


def write_model_result(output_dir, result):
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / 'metrics.json', result)
    write_json(output_dir / 'config.json', result.get('config', {}))
    return output_dir / 'metrics.json'


def skip_result(output_dir, model_spec, reason, config=None):
    result = {
        'experiment': model_spec['name'],
        'label': model_spec['label'],
        'description': model_spec.get('description', ''),
        'category': model_spec.get('category', ''),
        'status': 'skipped',
        'reason': reason,
        'config': config or {},
        'metrics': {},
    }
    write_model_result(output_dir, result)
    print(f"跳过模型 {model_spec['label']}: {reason}")
    return result


def finish_success(output_dir, model_spec, config, metrics, fit_seconds, predict_seconds):
    result = {
        'experiment': model_spec['name'],
        'label': model_spec['label'],
        'description': model_spec.get('description', ''),
        'category': model_spec.get('category', ''),
        'status': 'done',
        'fit_seconds': fit_seconds,
        'predict_seconds': predict_seconds,
        'config': config,
        'metrics': metrics,
    }
    write_model_result(output_dir, result)
    return result


def print_core_summary(rows):
    columns = ['label', 'mode', 'sample_f1', 'exact_match', 'micro_f1', 'macro_f1']
    if not rows:
        print('没有可展示的成功模型指标')
        return
    def fmt(value):
        if isinstance(value, float):
            return f'{value:.4f}'
        try:
            return f'{float(value):.4f}'
        except (TypeError, ValueError):
            return str(value)
    widths = {column: max(len(column), max(len(fmt(row.get(column, ''))) for row in rows)) for column in columns}
    header = ' | '.join(column.ljust(widths[column]) for column in columns)
    print('\n核心指标摘要')
    print(header)
    print('-' * len(header))
    for row in rows:
        print(' | '.join(fmt(row.get(column, '')).ljust(widths[column]) for column in columns))


def run_score_model(model_spec, args, score_builder, threshold_is_margin=False):
    output_dir = model_output_dir(model_spec['name'], run_id=args.run_id, output_dir=args.output_dir)
    logger = create_logger(output_dir / 'log.txt')
    config = vars(args).copy()
    config.update({
        'model_name': model_spec['name'],
        'label': model_spec['label'],
        'description': model_spec.get('description', ''),
        'threshold_is_margin': threshold_is_margin,
    })
    logger.info(f"开始对比模型: {model_spec['label']}")
    dataset = prepare_dataset(args.seed, args.test_ratio, args.max_se_num, args.data_path, output_dir, logger)
    env = dataset['env']
    x_train = vectorize_features(env, dataset['train_data'])
    y_train = vectorize_labels(env, dataset['train_data'])
    x_test = vectorize_features(env, dataset['test_data'])

    fit_start = time.perf_counter()
    scores = score_builder(x_train, y_train, x_test, dataset, logger)
    fit_predict_seconds = time.perf_counter() - fit_start
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != env.Se_action_num:
        raise ValueError(f'模型得分矩阵形状错误: {scores.shape}, 期望第二维={env.Se_action_num}')

    auto_pred = scores_to_actions(
        scores, 'auto', threshold=args.threshold, top_k=args.top_k,
        threshold_is_margin=threshold_is_margin
    )
    top2_pred = scores_to_actions(scores, 'top2', threshold=args.threshold, top_k=args.top_k)
    metrics = evaluate_contrast_predictions(env, dataset['test_data'], auto_pred, top2_pred, logger)
    result = finish_success(output_dir, model_spec, config, metrics, fit_predict_seconds, 0.0)
    logger.info(f"对比模型完成: {model_spec['label']}")
    print(f"模型完成: {model_spec['label']} -> {output_dir}")
    return result


def sklearn_or_skip(output_dir, model_spec, config=None):
    try:
        import sklearn  # noqa: F401
        return None
    except ImportError:
        return skip_result(output_dir, model_spec, '当前环境未安装scikit-learn，跳过该模型。', config=config)
