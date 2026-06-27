# -*- coding: utf-8 -*-
"""一键运行全部LHZ对比模型并汇总指标。"""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from common import RESULTS_DIR, current_timestamp, print_core_summary, write_batch_csvs, write_json

MODEL_REGISTRY = [
    {
        'name': 'frequency_prior',
        'label': 'Frequency Prior',
        'script': 'run_frequency_prior.py',
        'description': '训练集标签频率先验基线',
    },
    {
        'name': 'mlp_bce',
        'label': 'MLP-BCE',
        'script': 'run_mlp_bce.py',
        'description': '监督多标签神经网络基线',
    },
    {
        'name': 'logreg_ovr',
        'label': 'LogReg OvR',
        'script': 'run_logreg_ovr.py',
        'description': 'One-vs-Rest Logistic Regression',
    },
    {
        'name': 'bernoulli_nb',
        'label': 'Bernoulli NB',
        'script': 'run_bernoulli_nb.py',
        'description': 'One-vs-Rest Bernoulli Naive Bayes',
    },
    {
        'name': 'linear_svm',
        'label': 'Linear SVM OvR',
        'script': 'run_linear_svm.py',
        'description': 'One-vs-Rest Linear SVM',
    },
    {
        'name': 'random_forest',
        'label': 'Random Forest OvR',
        'script': 'run_random_forest.py',
        'description': 'One-vs-Rest Random Forest',
    },
    {
        'name': 'hist_gradient_boosting',
        'label': 'HistGradientBoosting OvR',
        'script': 'run_hist_gradient_boosting.py',
        'description': 'One-vs-Rest HistGradientBoosting',
    },
]
MODEL_BY_NAME = {item['name']: item for item in MODEL_REGISTRY}
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='一键运行全部或部分LHZ对比模型。')
    parser.add_argument('--models', nargs='+', default=['all'], choices=['all'] + [item['name'] for item in MODEL_REGISTRY])
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--max-se-num', type=int, default=0)
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--threshold', type=float, default=None, help='统一覆盖auto模式阈值；默认使用各模型自己的阈值')
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--mlp-epochs', type=int, default=30, help='传给MLP-BCE的训练轮数')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--hide-labels', action='store_true', help='保留兼容参数；当前仅影响终端输出，不影响CSV')
    return parser.parse_args()


def select_models(names):
    if 'all' in names:
        return MODEL_REGISTRY
    selected = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        selected.append(MODEL_BY_NAME[name])
        seen.add(name)
    return selected


def build_command(model, args, model_dir):
    command = [
        sys.executable,
        str(SCRIPT_DIR / model['script']),
        '--output-dir', str(model_dir),
        '--run-id', args.run_id,
        '--seed', str(args.seed),
        '--test-ratio', str(args.test_ratio),
        '--max-se-num', str(args.max_se_num),
        '--top-k', str(args.top_k),
    ]
    if args.threshold is not None:
        command.extend(['--threshold', str(args.threshold)])
    if args.data_path:
        command.extend(['--data-path', args.data_path])
    if model['name'] == 'mlp_bce':
        command.extend(['--epochs', str(args.mlp_epochs)])
    return command


def load_model_result(model_dir, model):
    metrics_path = model_dir / 'metrics.json'
    if not metrics_path.exists():
        return {
            'experiment': model['name'],
            'label': model['label'],
            'description': model['description'],
            'status': 'metrics_missing',
            'error': f'未找到metrics.json: {metrics_path}',
            'metrics': {},
        }
    return json.loads(metrics_path.read_text(encoding='utf-8'))


def run_model(model, args, batch_dir):
    model_dir = batch_dir / model['name']
    model_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(model, args, model_dir)
    log_path = model_dir / 'subprocess.log'
    if args.dry_run:
        print('[DRY-RUN] ' + ' '.join(command))
        return {
            'experiment': model['name'],
            'label': model['label'],
            'description': model['description'],
            'status': 'dry_run',
            'command': command,
            'output_dir': str(model_dir),
            'metrics': {},
        }

    print(f'========== 开始模型: {model["label"]} ==========')
    completed = subprocess.run(command, text=True, capture_output=True)
    log_path.write_text(
        'COMMAND:\n' + ' '.join(command) + '\n\nSTDOUT:\n' + completed.stdout + '\n\nSTDERR:\n' + completed.stderr,
        encoding='utf-8'
    )
    if completed.returncode != 0:
        return {
            'experiment': model['name'],
            'label': model['label'],
            'description': model['description'],
            'status': 'failed',
            'returncode': completed.returncode,
            'command': command,
            'log_file': str(log_path),
            'error': completed.stderr.strip() or completed.stdout.strip(),
            'metrics': {},
        }
    result = load_model_result(model_dir, model)
    result['command'] = command
    result['subprocess_log'] = str(log_path)
    print(f'========== 完成模型: {model["label"]} ({result.get("status")}) ==========')
    return result


def maybe_plot(batch_dir, args):
    if args.no_plot or args.dry_run:
        return None
    plot_script = SCRIPT_DIR / 'plot_contrast_metrics.py'
    command = [
        sys.executable,
        str(plot_script),
        '--result-dir', str(batch_dir),
        '--output', str(batch_dir / 'contrast_metrics.pdf'),
        '--no-summary',
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (batch_dir / 'plot.log').write_text(
        'COMMAND:\n' + ' '.join(command) + '\n\nSTDOUT:\n' + completed.stdout + '\n\nSTDERR:\n' + completed.stderr,
        encoding='utf-8'
    )
    if completed.returncode != 0:
        print('可视化生成失败，详情见 plot.log', file=sys.stderr)
        return None
    return str(batch_dir / 'contrast_metrics.pdf')


def main():
    args = parse_args()
    args.run_id = args.run_id or current_timestamp()
    batch_dir = RESULTS_DIR / args.run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    selected = select_models(args.models)

    print(f'结果批次目录: {batch_dir}')
    print('模型顺序: ' + ', '.join(item['label'] for item in selected))

    results = []
    for model in selected:
        try:
            result = run_model(model, args, batch_dir)
        except Exception as exc:
            result = {
                'experiment': model['name'],
                'label': model['label'],
                'description': model['description'],
                'status': 'failed',
                'error': str(exc),
                'traceback': traceback.format_exc(),
                'metrics': {},
            }
        results.append(result)
        if result.get('status') == 'failed' and not args.continue_on_error:
            break

    write_json(batch_dir / 'all_metrics.json', results)
    if not args.dry_run:
        aggregate_rows, label_rows = write_batch_csvs(batch_dir, results)
        print_core_summary(aggregate_rows)
        pdf_path = maybe_plot(batch_dir, args)
        print('\n指标文件已保存:')
        print(f'- JSON: {batch_dir / "all_metrics.json"}')
        print(f'- 汇总CSV: {batch_dir / "aggregate_metrics.csv"}')
        print(f'- 逐标签CSV: {batch_dir / "label_metrics.csv"}')
        if pdf_path:
            print(f'- 可视化PDF: {pdf_path}')
    else:
        print(f'Dry-run配置已保存: {batch_dir / "all_metrics.json"}')

    if any(item.get('status') == 'failed' for item in results):
        return 1
    if any(item.get('status') == 'metrics_missing' for item in results):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
