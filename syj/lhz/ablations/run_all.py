# -*- coding: utf-8 -*-
"""一键运行全部LHZ消融实验并汇总指标。"""

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path

from ablation_utils import EXPERIMENTS, RESULTS_DIR, current_timestamp, run_experiment


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

MODE_TITLES = {
    'auto': '测试集-模型自主停止',
}


def parse_float(value):
    return float(value)


def set_metric(metrics, key, value):
    metrics[key] = parse_float(value)


def find_log_file(output_dir):
    logs = sorted(Path(output_dir).glob('*.log'), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_metrics_from_log(log_path):
    """Parse aggregate and label-level metrics emitted by syj/lhz/mymodel/evaluation.py."""
    result = {
        'auto': {'metrics': {}, 'labels': {}},
    }
    current_mode = None

    core_pattern = re.compile(
        r'sample_f1:([0-9.]+), sample_recall:([0-9.]+), '
        r'sample_precision:([0-9.]+), exact_match:([0-9.]+), macro_f1:([0-9.]+)'
    )
    jaccard_pattern = re.compile(r'sample_jaccard\) avg:([0-9.]+) max:([0-9.]+) min:([0-9.]+)')
    exact_pattern = re.compile(r'exact_match\):([0-9.]+), 标签级Accuracy\(label_accuracy\):([0-9.]+), HammingLoss\(hamming_loss\):([0-9.]+)')
    micro_pattern = re.compile(r'Micro P/R/F1:([0-9.]+)/([0-9.]+)/([0-9.]+)')
    macro_pattern = re.compile(r'Macro P/R/F1:([0-9.]+)/([0-9.]+)/([0-9.]+)')
    hit_pattern = re.compile(r'HitRate\(hit_rate\):([0-9.]+), EmptyPredictionRate\(empty_prediction_rate\):([0-9.]+)')
    count_pattern = re.compile(
        r'平均推荐数:([0-9.]+), 平均真实数:([0-9.]+), 平均数量误差:([0-9.]+), '
        r'平均多选数:([0-9.]+), 平均漏选数:([0-9.]+)'
    )
    label_pattern = re.compile(
        r'\*\*标签\[(.+?)\] P/R/F1/Acc/Support/PredCount:'
        r'([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/(\d+)/(\d+)'
    )

    for raw_line in Path(log_path).read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if MODE_TITLES['auto'] in line:
            current_mode = 'auto'
            continue
        if current_mode is None:
            continue

        metrics = result[current_mode]['metrics']
        labels = result[current_mode]['labels']

        match = core_pattern.search(line)
        if match:
            set_metric(metrics, 'sample_f1', match.group(1))
            set_metric(metrics, 'sample_recall', match.group(2))
            set_metric(metrics, 'sample_precision', match.group(3))
            set_metric(metrics, 'exact_match', match.group(4))
            set_metric(metrics, 'macro_f1', match.group(5))
            continue

        match = jaccard_pattern.search(line)
        if match:
            set_metric(metrics, 'sample_jaccard', match.group(1))
            set_metric(metrics, 'sample_jaccard_max', match.group(2))
            set_metric(metrics, 'sample_jaccard_min', match.group(3))
            continue

        match = exact_pattern.search(line)
        if match:
            set_metric(metrics, 'exact_match', match.group(1))
            set_metric(metrics, 'label_accuracy', match.group(2))
            set_metric(metrics, 'hamming_loss', match.group(3))
            continue

        match = micro_pattern.search(line)
        if match:
            set_metric(metrics, 'micro_precision', match.group(1))
            set_metric(metrics, 'micro_recall', match.group(2))
            set_metric(metrics, 'micro_f1', match.group(3))
            continue

        match = macro_pattern.search(line)
        if match:
            set_metric(metrics, 'macro_precision', match.group(1))
            set_metric(metrics, 'macro_recall', match.group(2))
            set_metric(metrics, 'macro_f1', match.group(3))
            continue

        match = hit_pattern.search(line)
        if match:
            set_metric(metrics, 'hit_rate', match.group(1))
            set_metric(metrics, 'empty_prediction_rate', match.group(2))
            continue

        match = count_pattern.search(line)
        if match:
            set_metric(metrics, 'avg_selected_count', match.group(1))
            set_metric(metrics, 'avg_true_count', match.group(2))
            set_metric(metrics, 'avg_cardinality_error', match.group(3))
            set_metric(metrics, 'avg_over_select', match.group(4))
            set_metric(metrics, 'avg_under_select', match.group(5))
            continue

        match = label_pattern.search(line)
        if match:
            labels[match.group(1)] = {
                'precision': parse_float(match.group(2)),
                'recall': parse_float(match.group(3)),
                'f1': parse_float(match.group(4)),
                'accuracy': parse_float(match.group(5)),
                'support': int(match.group(6)),
                'pred_count': int(match.group(7)),
            }

    return result


def build_rows(results):
    aggregate_rows = []
    label_rows = []
    for item in results:
        experiment = item['experiment']
        label = item['label']
        parsed = item.get('metrics') or {}
        mode = 'auto'
        mode_data = parsed.get(mode, {})
        metrics = mode_data.get('metrics', {})
        row = {'experiment': experiment, 'label': label, 'mode': mode}
        for column in AGGREGATE_COLUMNS:
            if column not in row:
                row[column] = metrics.get(column, '')
        aggregate_rows.append(row)

        for syndrome_element, label_metrics in sorted(mode_data.get('labels', {}).items()):
            label_row = {
                'experiment': experiment,
                'label': label,
                'mode': mode,
                'syndrome_element': syndrome_element,
            }
            label_row.update(label_metrics)
            label_rows.append(label_row)
    return aggregate_rows, label_rows


def write_csv(path, rows, columns):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in columns})


def format_value(value):
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def print_table(rows, columns, title):
    print(f'\n{title}')
    if not rows:
        print('(无数据)')
        return
    widths = {}
    for column in columns:
        widths[column] = max(len(column), max(len(format_value(row.get(column, ''))) for row in rows))
    header = ' | '.join(column.ljust(widths[column]) for column in columns)
    print(header)
    print('-' * len(header))
    for row in rows:
        print(' | '.join(format_value(row.get(column, '')).ljust(widths[column]) for column in columns))


def parse_args():
    parser = argparse.ArgumentParser(description='一键运行全部或部分LHZ消融实验，并汇总展示指标。')
    parser.add_argument('--run-id', default=None, help='结果批次目录名；默认使用当前时间戳')
    parser.add_argument(
        '--experiments', nargs='+', default=['all'],
        choices=['all'] + [item['name'] for item in EXPERIMENTS],
        help='要运行的实验；默认all'
    )
    parser.add_argument('--dry-run', action='store_true', help='只打印将要执行的实验，不启动训练')
    parser.add_argument('--continue-on-error', action='store_true', help='某个实验失败后继续运行后续实验')
    parser.add_argument('--hide-labels', action='store_true', help='终端不展示逐标签指标；仍会保存到CSV/JSON')
    parser.add_argument(
        'main_args', nargs=argparse.REMAINDER,
        help='追加传给所有实验中syj/lhz/mymodel_reward_v2/main.py的参数；如需使用，请放在 -- 后面，例如: -- -episode 5'
    )
    args = parser.parse_args()
    if args.main_args and args.main_args[0] == '--':
        args.main_args = args.main_args[1:]
    return args


def select_experiments(names):
    if 'all' in names:
        return EXPERIMENTS
    selected = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        selected.append(next(item for item in EXPERIMENTS if item['name'] == name))
        seen.add(name)
    return selected


def main():
    args = parse_args()
    run_id = args.run_id or current_timestamp()
    batch_dir = RESULTS_DIR / run_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    selected = select_experiments(args.experiments)
    print(f'结果批次目录: {batch_dir}')
    print('实验顺序: ' + ', '.join(item['label'] for item in selected))
    if args.main_args:
        print('追加主程序参数: ' + ' '.join(args.main_args))

    results = []
    for experiment in selected:
        item = {
            'experiment': experiment['name'],
            'label': experiment['label'],
            'description': experiment['description'],
            'status': 'pending',
        }
        try:
            output_dir = run_experiment(
                experiment['name'],
                run_id=run_id,
                dry_run=args.dry_run,
                extra_main_args=args.main_args,
            )
            item['output_dir'] = str(output_dir)
            if args.dry_run:
                item['status'] = 'dry_run'
            else:
                log_file = find_log_file(output_dir)
                item['log_file'] = str(log_file) if log_file else ''
                if log_file is None:
                    item['status'] = 'metrics_missing'
                    item['error'] = '未找到日志文件，无法解析指标。'
                else:
                    item['metrics'] = parse_metrics_from_log(log_file)
                    item['status'] = 'done'
        except Exception as exc:
            item['status'] = 'failed'
            item['error'] = str(exc)
            item['traceback'] = traceback.format_exc()
            print(f'实验失败: {experiment["label"]}: {exc}', file=sys.stderr)
            if not args.continue_on_error:
                results.append(item)
                break
        results.append(item)

    summary_json = batch_dir / 'all_metrics.json'
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    if not args.dry_run:
        done_results = [item for item in results if item.get('status') == 'done']
        aggregate_rows, label_rows = build_rows(done_results)
        aggregate_csv = batch_dir / 'aggregate_metrics.csv'
        label_csv = batch_dir / 'label_metrics.csv'
        write_csv(aggregate_csv, aggregate_rows, AGGREGATE_COLUMNS)
        write_csv(label_csv, label_rows, LABEL_COLUMNS)

        print_table(aggregate_rows, AGGREGATE_COLUMNS, '汇总指标（实验 × 评估方式）')
        if not args.hide_labels:
            print_table(label_rows, LABEL_COLUMNS, '逐标签指标')

        print('\n指标文件已保存:')
        print(f'- JSON: {summary_json}')
        print(f'- 汇总CSV: {aggregate_csv}')
        print(f'- 逐标签CSV: {label_csv}')
    else:
        print(f'\nDry-run配置已保存: {summary_json}')

    failed = [item for item in results if item.get('status') == 'failed']
    if failed:
        return 1
    missing = [item for item in results if item.get('status') == 'metrics_missing']
    if missing:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
