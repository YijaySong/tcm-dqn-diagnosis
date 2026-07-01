# -*- coding: utf-8 -*-
"""LHZ对比实验指标可视化。"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError as exc:
    raise SystemExit('缺少matplotlib，无法绘图。请先安装: pip install matplotlib') from exc

SCRIPT_DIR = Path(__file__).resolve().parent
LHZ_DIR = SCRIPT_DIR.parent
MYMODEL_DIR = LHZ_DIR / 'mymodel'
RESULTS_DIR = SCRIPT_DIR / 'results'
OWN_MODEL_EXPERIMENT = 'our_rl_model'
OWN_MODEL_LABEL = 'Our RL Model'

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CORE_METRICS = [
    ('sample_f1', 'Sample F1'),
    ('sample_recall', 'Sample Recall'),
    ('sample_precision', 'Sample Precision'),
    ('exact_match', 'Exact Match'),
    ('micro_f1', 'Micro F1'),
    ('macro_f1', 'Macro F1'),
]

AUX_METRICS = [
    ('sample_jaccard', 'Sample Jaccard', '集合重合度，越高越好'),
    ('label_accuracy', 'Label Accuracy', '标签位准确率，标签稀疏时仅辅助参考'),
    ('hamming_loss', 'Hamming Loss', '标签位错误率，越低越好'),
    ('hit_rate', 'Hit Rate', '至少命中一个真实证候要素比例，越高越好'),
    ('empty_prediction_rate', 'Empty Prediction Rate', '空预测比例，通常越低越好'),
    ('avg_selected_count', 'Avg Selected Count', '平均推荐数量'),
    ('avg_true_count', 'Avg True Count', '平均真实数量'),
    ('avg_cardinality_error', 'Avg Count Error', '推荐数量误差，越低越好'),
    ('avg_over_select', 'Avg Over-select', '平均多选数，越低越好'),
    ('avg_under_select', 'Avg Under-select', '平均漏选数，越低越好'),
]

MODEL_ORDER = [
    'frequency_prior',
    'mlp_bce',
    'logreg_ovr',
    'bernoulli_nb',
    'linear_svm',
    'random_forest',
    'hist_gradient_boosting',
    OWN_MODEL_EXPERIMENT,
]
MODE_ORDER = ['auto', 'top2']


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as file_obj:
        return list(csv.DictReader(file_obj))


def read_json(path):
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding='utf-8'))


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    return float(value)


def format_value(value):
    if value is None or value == '':
        return '-'
    try:
        return f'{float(value):.4f}'
    except (TypeError, ValueError):
        return str(value)


def latest_result_dir():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f'未找到结果目录: {RESULTS_DIR}')
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f'结果目录为空: {RESULTS_DIR}')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_result_dir(result_dir):
    result_dir = Path(result_dir)
    aggregate_path = result_dir / 'aggregate_metrics.csv'
    label_path = result_dir / 'label_metrics.csv'
    if not aggregate_path.exists():
        raise FileNotFoundError(f'未找到汇总指标文件: {aggregate_path}')
    if not label_path.exists():
        raise FileNotFoundError(f'未找到逐标签指标文件: {label_path}')
    return read_csv(aggregate_path), read_csv(label_path), read_json(result_dir / 'all_metrics.json')


def find_latest_own_model_log():
    """优先查找完整模型日志，最后才回退到任意DQN日志。"""
    project_root = LHZ_DIR.parent.parent
    search_patterns = [
        str(MYMODEL_DIR / 'dqn_*.log'),
        str(Path.cwd() / 'dqn_*.log'),
        str(project_root / 'dqn_*.log'),
        str(LHZ_DIR / 'dqn_*.log'),
        str(LHZ_DIR / 'ablations' / 'results' / '*' / 'full' / 'dqn_*.log'),
        str(LHZ_DIR / 'ablations' / 'results' / '*' / 'full' / '**' / 'dqn_*.log'),
        str(LHZ_DIR / 'ablations' / 'results' / '**' / 'dqn_*.log'),
    ]
    for pattern in search_patterns:
        recursive = '**' in pattern
        matches = [Path(path) for path in glob.glob(pattern, recursive=recursive) if Path(path).exists()]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
    return None


def parse_own_model_log(log_path):
    metrics = {'auto': {}, 'top2': {}}
    current_mode = None
    core_pattern = re.compile(
        r'sample_f1:([0-9.]+), sample_recall:([0-9.]+), '
        r'sample_precision:([0-9.]+), exact_match:([0-9.]+), macro_f1:([0-9.]+)'
    )
    jaccard_pattern = re.compile(r'sample_jaccard\) avg:([0-9.]+)')
    exact_pattern = re.compile(r'exact_match\):([0-9.]+), 标签级Accuracy\(label_accuracy\):([0-9.]+), HammingLoss\(hamming_loss\):([0-9.]+)')
    micro_pattern = re.compile(r'Micro P/R/F1:([0-9.]+)/([0-9.]+)/([0-9.]+)')
    macro_pattern = re.compile(r'Macro P/R/F1:([0-9.]+)/([0-9.]+)/([0-9.]+)')
    hit_pattern = re.compile(r'HitRate\(hit_rate\):([0-9.]+), EmptyPredictionRate\(empty_prediction_rate\):([0-9.]+)')
    count_pattern = re.compile(
        r'平均推荐数:([0-9.]+), 平均真实数:([0-9.]+), 平均数量误差:([0-9.]+), '
        r'平均多选数:([0-9.]+), 平均漏选数:([0-9.]+)'
    )
    final_auto_pattern = re.compile(r'测试集-自主停止: sample_f1=([0-9.]+), exact_match=([0-9.]+), micro_f1=([0-9.]+), macro_f1=([0-9.]+)')
    final_top2_pattern = re.compile(r'测试集-Top-2: sample_f1=([0-9.]+), exact_match=([0-9.]+), micro_f1=([0-9.]+), macro_f1=([0-9.]+)')

    for raw_line in Path(log_path).read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if '**测试集-模型自主停止' in line or '**模型自主停止' in line:
            current_mode = 'auto'
            continue
        if '**测试集-固定Top-2诊断' in line or '**固定Top-2诊断' in line:
            current_mode = 'top2'
            continue

        match = final_auto_pattern.search(line)
        if match:
            metrics['auto'].update({
                'sample_f1': float(match.group(1)),
                'exact_match': float(match.group(2)),
                'micro_f1': float(match.group(3)),
                'macro_f1': float(match.group(4)),
            })
            continue
        match = final_top2_pattern.search(line)
        if match:
            metrics['top2'].update({
                'sample_f1': float(match.group(1)),
                'exact_match': float(match.group(2)),
                'micro_f1': float(match.group(3)),
                'macro_f1': float(match.group(4)),
            })
            continue

        if current_mode is None:
            continue
        target = metrics[current_mode]
        match = core_pattern.search(line)
        if match:
            target.update({
                'sample_f1': float(match.group(1)),
                'sample_recall': float(match.group(2)),
                'sample_precision': float(match.group(3)),
                'exact_match': float(match.group(4)),
                'macro_f1': float(match.group(5)),
            })
            continue
        match = jaccard_pattern.search(line)
        if match:
            target['sample_jaccard'] = float(match.group(1))
            continue
        match = exact_pattern.search(line)
        if match:
            target['exact_match'] = float(match.group(1))
            target['label_accuracy'] = float(match.group(2))
            target['hamming_loss'] = float(match.group(3))
            continue
        match = micro_pattern.search(line)
        if match:
            target['micro_precision'] = float(match.group(1))
            target['micro_recall'] = float(match.group(2))
            target['micro_f1'] = float(match.group(3))
            continue
        match = macro_pattern.search(line)
        if match:
            target['macro_precision'] = float(match.group(1))
            target['macro_recall'] = float(match.group(2))
            target['macro_f1'] = float(match.group(3))
            continue
        match = hit_pattern.search(line)
        if match:
            target['hit_rate'] = float(match.group(1))
            target['empty_prediction_rate'] = float(match.group(2))
            continue
        match = count_pattern.search(line)
        if match:
            target['avg_selected_count'] = float(match.group(1))
            target['avg_true_count'] = float(match.group(2))
            target['avg_cardinality_error'] = float(match.group(3))
            target['avg_over_select'] = float(match.group(4))
            target['avg_under_select'] = float(match.group(5))
    return metrics


def append_own_model_rows(aggregate_rows, results, log_path=None):
    log_path = Path(log_path) if log_path else find_latest_own_model_log()
    if log_path is None:
        print('未找到我们自己模型的DQN日志，contrast图中不会加入Our RL Model。')
        return aggregate_rows, results
    metrics = parse_own_model_log(log_path)
    aggregate_rows = [row for row in aggregate_rows if row.get('experiment') != OWN_MODEL_EXPERIMENT]
    for mode in MODE_ORDER:
        row = {
            'experiment': OWN_MODEL_EXPERIMENT,
            'label': OWN_MODEL_LABEL,
            'mode': mode,
        }
        row.update(metrics.get(mode, {}))
        aggregate_rows.append(row)
    results = [item for item in results if item.get('experiment') != OWN_MODEL_EXPERIMENT]
    results.append({
        'experiment': OWN_MODEL_EXPERIMENT,
        'label': OWN_MODEL_LABEL,
        'status': 'from_log',
        'reason': str(log_path),
    })
    print(f'已加入我们自己模型指标: {log_path}')
    return aggregate_rows, results


def sort_rows(rows):
    model_rank = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    mode_rank = {name: idx for idx, name in enumerate(MODE_ORDER)}
    return sorted(rows, key=lambda row: (mode_rank.get(row.get('mode'), 99), model_rank.get(row.get('experiment'), 99), row.get('experiment', '')))


def add_table_page(title, headers, rows, figsize=(16, 9), font_size=8, scale_y=1.2):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    if not rows:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        return fig
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e9edf5')
        elif row_idx % 2 == 0:
            cell.set_facecolor('#f8f9fb')
    return fig


def plot_core_metrics(rows, mode, title_suffix):
    rows = sort_rows([row for row in rows if row.get('mode') == mode])
    labels = [row.get('label') or row.get('experiment') for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    for idx, (key, title) in enumerate(CORE_METRICS):
        ax = axes[idx]
        values = [to_float(row.get(key)) for row in rows]
        colors = ['#D62728' if row.get('experiment') == OWN_MODEL_EXPERIMENT else '#4C78A8' for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.02, 0.98), f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'Contrast Core Metrics - {title_suffix}', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_heatmap(rows):
    rows = sort_rows(rows)
    metric_keys = [key for key, _ in CORE_METRICS]
    metric_labels = [label for _, label in CORE_METRICS]
    row_labels = [f"{row.get('label')} ({row.get('mode')})" for row in rows]
    values = [[to_float(row.get(key)) for key in metric_keys] for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(5, len(rows) * 0.45)))
    image = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title('Contrast Core Metrics Heatmap', fontsize=14, fontweight='bold')
    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            ax.text(col_idx, row_idx, f'{value:.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def build_aux_rows(rows, mode):
    output = []
    for row in sort_rows([item for item in rows if item.get('mode') == mode]):
        label = row.get('label') or row.get('experiment')
        for key, name, note in AUX_METRICS:
            output.append([label, name, format_value(row.get(key)), note])
    return output


def build_label_rows(rows, mode, top_n):
    output = []
    grouped = {}
    for row in sort_rows([item for item in rows if item.get('mode') == mode]):
        grouped.setdefault(row.get('label') or row.get('experiment'), []).append(row)
    for label, items in grouped.items():
        ranked = sorted(items, key=lambda item: to_float(item.get('support')), reverse=True)
        for item in ranked[:top_n]:
            output.append([
                label,
                item.get('syndrome_element', ''),
                format_value(item.get('precision')),
                format_value(item.get('recall')),
                format_value(item.get('f1')),
                format_value(item.get('accuracy')),
                format_value(item.get('support')),
                format_value(item.get('pred_count')),
            ])
    return output


def build_status_rows(results):
    rows = []
    for item in results:
        rows.append([
            item.get('label', item.get('experiment', '')),
            item.get('status', ''),
            format_value(item.get('fit_seconds', '')),
            format_value(item.get('predict_seconds', '')),
            item.get('reason', item.get('error', '')),
        ])
    return rows


def make_figures(aggregate_rows, label_rows, results, top_n_labels):
    figures = []
    figures.append(plot_core_metrics(aggregate_rows, 'auto', 'Auto / Threshold'))
    figures.append(plot_core_metrics(aggregate_rows, 'top2', 'Top-2'))
    figures.append(plot_heatmap(aggregate_rows))
    figures.append(add_table_page('Model Status Table', ['Model', 'Status', 'Fit Seconds', 'Predict Seconds', 'Note'], build_status_rows(results), font_size=8))
    for mode, title in [('auto', 'Auto / Threshold'), ('top2', 'Top-2')]:
        figures.append(add_table_page(f'Auxiliary Metrics - {title}', ['Model', 'Metric', 'Value', 'Note'], build_aux_rows(aggregate_rows, mode), figsize=(16, 10), font_size=7, scale_y=1.1))
        figures.append(add_table_page(f'Top-{top_n_labels} Label Metrics - {title}', ['Model', 'Syndrome Element', 'Precision', 'Recall', 'F1', 'Accuracy', 'Support', 'Pred Count'], build_label_rows(label_rows, mode, top_n_labels), figsize=(18, 11), font_size=6, scale_y=0.95))
    return figures


def save_figures(figures, output_path, also_png=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != '.pdf':
        raise ValueError('输出文件必须是PDF，例如 contrast_metrics.pdf')
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
    print(f'可视化PDF已保存: {output_path}')
    if also_png:
        stem = output_path.with_suffix('')
        for idx, fig in enumerate(figures, start=1):
            png_path = Path(f'{stem}_page{idx}.png')
            fig.savefig(png_path, dpi=160, bbox_inches='tight')
            print(f'PNG第{idx}页已保存: {png_path}')
    for fig in figures:
        plt.close(fig)


def print_summary(rows):
    columns = ['label', 'mode', 'sample_f1', 'exact_match', 'micro_f1', 'macro_f1']
    rows = sort_rows(rows)
    widths = {column: max(len(column), max(len(format_value(row.get(column))) for row in rows)) for column in columns}
    print('\n核心指标摘要')
    header = ' | '.join(column.ljust(widths[column]) for column in columns)
    print(header)
    print('-' * len(header))
    for row in rows:
        print(' | '.join(format_value(row.get(column)).ljust(widths[column]) for column in columns))


def parse_args():
    parser = argparse.ArgumentParser(description='将LHZ对比实验指标可视化为图和表。')
    parser.add_argument('--result-dir', default=None, help='run_all.py生成的结果批次目录；默认读取最新目录')
    parser.add_argument('--output', default=None, help='输出PDF路径；默认保存为结果目录/contrast_metrics.pdf')
    parser.add_argument('--also-png', action='store_true')
    parser.add_argument('--top-n-labels', type=int, default=15)
    parser.add_argument('--own-log', default=None, help='我们自己DQN/RL模型的日志路径；默认自动查找最新dqn_*.log')
    parser.add_argument('--no-own-model', action='store_true', help='不把我们自己的DQN/RL模型加入对比图')
    parser.add_argument('--no-summary', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir) if args.result_dir else latest_result_dir()
    aggregate_rows, label_rows, results = load_result_dir(result_dir)
    if not args.no_own_model:
        aggregate_rows, results = append_own_model_rows(aggregate_rows, results, log_path=args.own_log)
    output = Path(args.output) if args.output else result_dir / 'contrast_metrics.pdf'
    figures = make_figures(aggregate_rows, label_rows, results, args.top_n_labels)
    save_figures(figures, output, also_png=args.also_png)
    if not args.no_summary:
        print_summary(aggregate_rows)
    print(f'读取结果目录: {result_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
