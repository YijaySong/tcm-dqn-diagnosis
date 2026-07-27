# -*- coding: utf-8 -*-
"""消融实验指标可视化。

读取`run_all.py`生成的`aggregate_metrics.csv`和`label_metrics.csv`：
- 主要指标用柱状图展示。
- 辅助指标用表格展示。
- 逐标签指标单独生成表格页。

默认读取`syj/lhz/ablations/results/`下最新的实验批次目录。
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError as exc:
    raise SystemExit(
        '缺少matplotlib，无法绘图。请先安装: pip install matplotlib'
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
LHZ_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / 'results'
IMPROVED_EXPERIMENT = 'reward_v2'
IMPROVED_LABEL = 'Reward V2'

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
    ('supported_macro_f1', 'Supported Macro F1', '仅对测试集中出现过的标签求Macro F1'),
    ('hit_rate', 'Hit Rate', '至少命中一个真实证候要素的比例，越高越好'),
    ('empty_prediction_rate', 'Empty Prediction Rate', '空预测比例，通常越低越好'),
    ('avg_selected_count', 'Avg Selected Count', '平均推荐证候要素数量'),
    ('avg_true_count', 'Avg True Count', '平均真实证候要素数量'),
    ('avg_cardinality_error', 'Avg Count Error', '推荐数量误差，越低越好'),
    ('avg_over_select', 'Avg Over-select', '平均多选数，越低越好'),
    ('avg_under_select', 'Avg Under-select', '平均漏选数，越低越好'),
]

LABEL_METRICS = [
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1', 'F1'),
    ('accuracy', 'Accuracy'),
    ('support', 'Support'),
    ('pred_count', 'Pred Count'),
]

EXPERIMENT_ORDER = ['full', 'no_pretrain', 'pretrain_only', 'no_expert_warmup', 'rl_only', IMPROVED_EXPERIMENT]
MODE_ORDER = ['auto']


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as file_obj:
        return list(csv.DictReader(file_obj))


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
    return read_csv(aggregate_path), read_csv(label_path)


def is_improved_model_log(log_path):
    try:
        text = Path(log_path).read_text(encoding='utf-8')[:12000]
    except OSError:
        return False
    return 'tail_cost_curiosity' in text or "'mode': 'tail_cost_curiosity'" in text


def find_latest_improved_log():
    project_root = LHZ_DIR.parent.parent
    patterns = [
        str(LHZ_DIR / 'mymodel_reward_v2' / 'dqn_*.log'),
        str(Path.cwd() / 'dqn_*.log'),
        str(project_root / 'dqn_*.log'),
        str(LHZ_DIR / 'dqn_*.log'),
        str(RESULTS_DIR / '**' / 'dqn_*.log'),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(Path(path) for path in glob.glob(pattern, recursive='**' in pattern) if Path(path).exists())
    matches = [path for path in set(matches) if is_improved_model_log(path)]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def parse_improved_log(log_path):
    parsed = {'auto': {'metrics': {}, 'labels': {}}}
    current_mode = None
    core_pattern = re.compile(
        r'sample_f1:([0-9.]+), sample_recall:([0-9.]+), '
        r'sample_precision:([0-9.]+), exact_match:([0-9.]+), macro_f1:([0-9.]+)'
    )
    jaccard_pattern = re.compile(r'sample_jaccard\) avg:([0-9.]+) max:([0-9.]+) min:([0-9.]+)')
    exact_pattern = re.compile(r'exact_match\):([0-9.]+), 标签级Accuracy\(label_accuracy\):([0-9.]+), HammingLoss\(hamming_loss\):([0-9.]+)')
    micro_pattern = re.compile(r'Micro P/R/F1:([0-9.]+)/([0-9.]+)/([0-9.]+)')
    macro_pattern = re.compile(r'Macro P/R/F1:([0-9.]+)/([0-9.]+)/([0-9.]+)')
    supported_macro_pattern = re.compile(r'Supported Macro P/R/F1:([0-9.]+)/([0-9.]+)/([0-9.]+)')
    hit_pattern = re.compile(r'HitRate\(hit_rate\):([0-9.]+), EmptyPredictionRate\(empty_prediction_rate\):([0-9.]+)')
    count_pattern = re.compile(
        r'平均推荐数:([0-9.]+), 平均真实数:([0-9.]+), 平均数量误差:([0-9.]+), '
        r'平均多选数:([0-9.]+), 平均漏选数:([0-9.]+)'
    )
    label_pattern = re.compile(
        r'\*\*标签\[(.+?)\] P/R/F1/Acc/Support/PredCount:'
        r'([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/(\d+)/(\d+)'
    )
    final_auto_pattern = re.compile(
        r'测试集-自主停止: sample_f1=([0-9.]+), exact_match=([0-9.]+), '
        r'micro_f1=([0-9.]+), macro_f1=([0-9.]+)(?:, supported_macro_f1=([0-9.]+))?'
    )

    for raw_line in Path(log_path).read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if '测试集-模型自主停止' in line or '模型自主停止' in line:
            current_mode = 'auto'
            continue

        match = final_auto_pattern.search(line)
        if match:
            metrics = parsed['auto']['metrics']
            metrics['sample_f1'] = float(match.group(1))
            metrics['exact_match'] = float(match.group(2))
            metrics['micro_f1'] = float(match.group(3))
            metrics['macro_f1'] = float(match.group(4))
            if match.group(5) is not None:
                metrics['supported_macro_f1'] = float(match.group(5))
            continue

        if current_mode is None:
            continue
        metrics = parsed[current_mode]['metrics']
        labels = parsed[current_mode]['labels']

        match = core_pattern.search(line)
        if match:
            metrics['sample_f1'] = float(match.group(1))
            metrics['sample_recall'] = float(match.group(2))
            metrics['sample_precision'] = float(match.group(3))
            metrics['exact_match'] = float(match.group(4))
            metrics['macro_f1'] = float(match.group(5))
            continue
        match = jaccard_pattern.search(line)
        if match:
            metrics['sample_jaccard'] = float(match.group(1))
            metrics['sample_jaccard_max'] = float(match.group(2))
            metrics['sample_jaccard_min'] = float(match.group(3))
            continue
        match = exact_pattern.search(line)
        if match:
            metrics['exact_match'] = float(match.group(1))
            metrics['label_accuracy'] = float(match.group(2))
            metrics['hamming_loss'] = float(match.group(3))
            continue
        match = micro_pattern.search(line)
        if match:
            metrics['micro_precision'] = float(match.group(1))
            metrics['micro_recall'] = float(match.group(2))
            metrics['micro_f1'] = float(match.group(3))
            continue
        match = macro_pattern.search(line)
        if match:
            metrics['macro_precision'] = float(match.group(1))
            metrics['macro_recall'] = float(match.group(2))
            metrics['macro_f1'] = float(match.group(3))
            continue
        match = supported_macro_pattern.search(line)
        if match:
            metrics['supported_macro_precision'] = float(match.group(1))
            metrics['supported_macro_recall'] = float(match.group(2))
            metrics['supported_macro_f1'] = float(match.group(3))
            continue
        match = hit_pattern.search(line)
        if match:
            metrics['hit_rate'] = float(match.group(1))
            metrics['empty_prediction_rate'] = float(match.group(2))
            continue
        match = count_pattern.search(line)
        if match:
            metrics['avg_selected_count'] = float(match.group(1))
            metrics['avg_true_count'] = float(match.group(2))
            metrics['avg_cardinality_error'] = float(match.group(3))
            metrics['avg_over_select'] = float(match.group(4))
            metrics['avg_under_select'] = float(match.group(5))
            continue
        match = label_pattern.search(line)
        if match:
            labels[match.group(1)] = {
                'precision': float(match.group(2)),
                'recall': float(match.group(3)),
                'f1': float(match.group(4)),
                'accuracy': float(match.group(5)),
                'support': int(match.group(6)),
                'pred_count': int(match.group(7)),
            }
    return parsed


def append_improved_model_rows(aggregate_rows, label_rows, log_path=None):
    log_path = Path(log_path) if log_path else find_latest_improved_log()
    if log_path is None:
        print('未找到Reward V2日志，消融图中不会加入该模型。')
        return aggregate_rows, label_rows

    parsed = parse_improved_log(log_path)
    aggregate_rows = [row for row in aggregate_rows if row.get('experiment') != IMPROVED_EXPERIMENT]
    label_rows = [row for row in label_rows if row.get('experiment') != IMPROVED_EXPERIMENT]
    for mode in MODE_ORDER:
        metrics = parsed.get(mode, {}).get('metrics', {})
        row = {'experiment': IMPROVED_EXPERIMENT, 'label': IMPROVED_LABEL, 'mode': mode}
        row.update(metrics)
        aggregate_rows.append(row)
        for syndrome_element, metrics_row in sorted(parsed.get(mode, {}).get('labels', {}).items()):
            label_row = {
                'experiment': IMPROVED_EXPERIMENT,
                'label': IMPROVED_LABEL,
                'mode': mode,
                'syndrome_element': syndrome_element,
            }
            label_row.update(metrics_row)
            label_rows.append(label_row)
    print(f'已加入Reward V2指标: {log_path}')
    return aggregate_rows, label_rows


def sort_aggregate_rows(rows):
    experiment_rank = {name: idx for idx, name in enumerate(EXPERIMENT_ORDER)}
    mode_rank = {name: idx for idx, name in enumerate(MODE_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            mode_rank.get(row.get('mode'), 999),
            experiment_rank.get(row.get('experiment'), 999),
            row.get('experiment', ''),
        )
    )


def sort_label_rows(rows):
    experiment_rank = {name: idx for idx, name in enumerate(EXPERIMENT_ORDER)}
    mode_rank = {name: idx for idx, name in enumerate(MODE_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            mode_rank.get(row.get('mode'), 999),
            experiment_rank.get(row.get('experiment'), 999),
            row.get('syndrome_element', ''),
        )
    )


def add_table_page(title, headers, rows, figsize=(16, 9), font_size=8, scale_y=1.25):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    if not rows:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        return fig

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
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


def plot_core_metrics(aggregate_rows, mode, title_suffix):
    rows = [row for row in aggregate_rows if row.get('mode') == mode]
    rows = sort_aggregate_rows(rows)
    labels = [row.get('label') or row.get('experiment') for row in rows]
    metric_keys = [key for key, _ in CORE_METRICS]
    metric_labels = [label for _, label in CORE_METRICS]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.flatten()
    for idx, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[idx]
        values = [to_float(row.get(key)) for row in rows]
        colors = ['#D62728' if row.get('experiment') == IMPROVED_EXPERIMENT else '#4C78A8' for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.02, 0.98),
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
    fig.suptitle(f'Core Metrics - {title_suffix}', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_core_heatmap(aggregate_rows):
    rows = sort_aggregate_rows(aggregate_rows)
    row_labels = [f"{row.get('label')} ({row.get('mode')})" for row in rows]
    metric_keys = [key for key, _ in CORE_METRICS]
    metric_labels = [label for _, label in CORE_METRICS]
    values = [[to_float(row.get(key)) for key in metric_keys] for row in rows]

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(rows))))
    image = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title('Core Metrics Heatmap', fontsize=14, fontweight='bold')
    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            ax.text(col_idx, row_idx, f'{value:.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def build_aux_table_rows(aggregate_rows, mode):
    rows = [row for row in sort_aggregate_rows(aggregate_rows) if row.get('mode') == mode]
    table_rows = []
    for row in rows:
        prefix = row.get('label') or row.get('experiment')
        for key, label, note in AUX_METRICS:
            table_rows.append([prefix, label, format_value(row.get(key)), note])
    return table_rows


def build_label_table_rows(label_rows, mode, top_n):
    rows = [row for row in sort_label_rows(label_rows) if row.get('mode') == mode]
    output = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get('label') or row.get('experiment'), []).append(row)

    for experiment_label, experiment_rows in grouped.items():
        ranked = sorted(experiment_rows, key=lambda row: to_float(row.get('support')), reverse=True)
        for row in ranked[:top_n]:
            output.append([
                experiment_label,
                row.get('syndrome_element', ''),
                format_value(row.get('precision')),
                format_value(row.get('recall')),
                format_value(row.get('f1')),
                format_value(row.get('accuracy')),
                format_value(row.get('support')),
                format_value(row.get('pred_count')),
            ])
    return output


def make_figures(aggregate_rows, label_rows, top_n_labels):
    figures = []
    if not aggregate_rows:
        figures.append(add_table_page('Ablation Metrics', ['Message'], [['No aggregate metrics found']]))
        return figures

    figures.append(plot_core_metrics(aggregate_rows, 'auto', 'Auto Stop'))
    figures.append(plot_core_heatmap(aggregate_rows))

    aux_rows = build_aux_table_rows(aggregate_rows, 'auto')
    figures.append(add_table_page(
        'Auxiliary Metrics Table - Auto Stop',
        ['Experiment', 'Metric', 'Value', 'Note'],
        aux_rows,
        figsize=(16, 10),
        font_size=7,
        scale_y=1.1,
    ))

    if label_rows:
        table_rows = build_label_table_rows(label_rows, 'auto', top_n_labels)
        figures.append(add_table_page(
            f'Top-{top_n_labels} Label Metrics by Support - Auto Stop',
            ['Experiment', 'Syndrome Element', 'Precision', 'Recall', 'F1', 'Accuracy', 'Support', 'Pred Count'],
            table_rows,
            figsize=(18, 11),
            font_size=6,
            scale_y=0.95,
        ))
    return figures


def save_figures(figures, output_path, also_png=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext != '.pdf':
        raise ValueError('当前脚本主输出请使用.pdf后缀，例如 ablation_metrics.pdf')

    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
    print(f'可视化PDF已保存: {output_path}')

    if also_png:
        stem = output_path.with_suffix('')
        for idx, fig in enumerate(figures, start=1):
            page_path = Path(f'{stem}_page{idx}.png')
            fig.savefig(page_path, dpi=160, bbox_inches='tight')
            print(f'PNG第{idx}页已保存: {page_path}')

    for fig in figures:
        plt.close(fig)


def print_brief_summary(aggregate_rows):
    rows = sort_aggregate_rows(aggregate_rows)
    columns = ['label', 'mode', 'sample_f1', 'exact_match', 'micro_f1', 'macro_f1']
    widths = {}
    for column in columns:
        widths[column] = max(len(column), max(len(format_value(row.get(column))) for row in rows))
    header = ' | '.join(column.ljust(widths[column]) for column in columns)
    print('\n核心指标摘要')
    print(header)
    print('-' * len(header))
    for row in rows:
        print(' | '.join(format_value(row.get(column)).ljust(widths[column]) for column in columns))


def parse_args():
    parser = argparse.ArgumentParser(description='将LHZ消融实验指标可视化为图和表。')
    parser.add_argument(
        '--result-dir', default=None,
        help='run_all.py生成的某个结果批次目录；默认选择syj/lhz/ablations/results下最新目录'
    )
    parser.add_argument('--output', default=None, help='输出PDF路径；默认保存到结果批次目录/ablation_metrics.pdf')
    parser.add_argument('--also-png', action='store_true', help='除PDF外，同时把每页保存为PNG')
    parser.add_argument('--top-n-labels', type=int, default=15, help='逐标签表格中每个实验展示support最高的前N个标签')
    parser.add_argument('--reward-v2-log', default=None, help='Reward V2模型日志；默认自动查找最新tail_cost_curiosity日志')
    parser.add_argument('--no-reward-v2', action='store_true', help='不把Reward V2模型加入消融对比图')
    parser.add_argument('--no-summary', action='store_true', help='不在终端打印核心指标摘要')
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir) if args.result_dir else latest_result_dir()
    aggregate_rows, label_rows = load_result_dir(result_dir)
    if not args.no_reward_v2:
        aggregate_rows, label_rows = append_improved_model_rows(
            aggregate_rows, label_rows, log_path=args.reward_v2_log
        )
    output_path = Path(args.output) if args.output else result_dir / 'ablation_metrics.pdf'

    figures = make_figures(aggregate_rows, label_rows, args.top_n_labels)
    save_figures(figures, output_path, also_png=args.also_png)
    if not args.no_summary:
        print_brief_summary(aggregate_rows)
    print(f'读取结果目录: {result_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
