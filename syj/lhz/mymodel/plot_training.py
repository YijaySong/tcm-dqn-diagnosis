# -*- coding: utf-8 -*-
"""训练日志绘图文件。

负责解析DQN训练过程中生成的日志文件，提取reward、loss、动作统计、课程学习
阶段和测试集评估指标。输出时既展示最终测试结果，也展示训练收敛曲线、平稳性
诊断和辅助指标表格。默认保存为多页PDF，便于分开展示核心图表和辅助信息。
"""
import os
import argparse
import glob
import re
import sys

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lhz_matplotlib_cache')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CORE_METRICS = [
    ('f1', 'Sample F1'),
    ('recall', 'Sample Recall'),
    ('precision', 'Sample Precision'),
    ('exact_match', 'Exact Match'),
    ('macro_f1', 'Macro F1'),
]

COMPARISON_METRICS = CORE_METRICS + [
    ('micro_f1', 'Micro F1'),
    ('supported_macro_f1', 'Supported Macro F1'),
]

AUX_METRICS = [
    ('jaccard', 'Sample Jaccard'),
    ('label_accuracy', 'Label Accuracy'),
    ('hamming_loss', 'Hamming Loss'),
    ('micro_f1', 'Micro F1'),
    ('hit_rate', 'Hit Rate'),
    ('empty_prediction_rate', 'Empty Prediction Rate'),
    ('avg_selected_count', 'Avg Selected Count'),
    ('avg_true_count', 'Avg True Count'),
    ('avg_cardinality_error', 'Avg Count Error'),
    ('avg_over_select', 'Avg Over-select'),
    ('avg_under_select', 'Avg Under-select'),
    ('false_selection_rate', 'False-selection Rate'),
    ('miss_selection_rate', 'Miss-selection Rate'),
]

SMOOTH_WINDOW = 10
CURRICULUM_COLORS = {
    'len<=2': '#4C78A8',
    'len<=4': '#59A14F',
    'all': '#F58518',
}


def parse_log(log_path):
    """解析日志文件。

    兼容两种日志：
    1. 旧日志：每个 episode 后都有模型自主停止评估。
    2. 新日志：episode 内只记录训练 reward/loss，训练完成后单独记录测试集指标。
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    episodes = []
    final_metrics = {'auto': {}}
    pending_ep = None
    current_section = None
    auto_metrics = {}
    collecting_final = False

    def commit_episode():
        nonlocal pending_ep, auto_metrics
        if pending_ep is not None:
            pending_ep['auto'] = auto_metrics.copy()
            episodes.append(pending_ep)
        pending_ep = None
        auto_metrics = {}

    def parse_metric_line(line, target):
        m = re.search(
            r'\*\*核心指标 sample_f1:([\d.]+), sample_recall:([\d.]+), '
            r'sample_precision:([\d.]+), exact_match:([\d.]+), macro_f1:([\d.]+)',
            line
        )
        if m:
            target['f1'] = float(m.group(1))
            target['sample_f1'] = float(m.group(1))
            target['recall'] = float(m.group(2))
            target['precision'] = float(m.group(3))
            target['exact_match'] = float(m.group(4))
            target['macro_f1'] = float(m.group(5))

        m = re.search(r'\*\*样本Jaccard(?:/集合准确率|\(sample_jaccard\))? avg:([\d.]+)', line)
        if m:
            target['jaccard'] = float(m.group(1))

        m = re.search(r'\*\*样本Precision avg:([\d.]+), Recall avg:([\d.]+), F1 avg:([\d.]+)', line)
        if m:
            target['precision'] = float(m.group(1))
            target['recall'] = float(m.group(2))
            target['f1'] = float(m.group(3))
            target['sample_f1'] = float(m.group(3))

        m = re.search(r'\*\*ExactMatch(?:/子集准确率|/严格准确率)?(?:\(exact_match\))?:([\d.]+).*HammingLoss(?:\(hamming_loss\))?:([\d.]+)', line)
        if m:
            target['exact_match'] = float(m.group(1))
            target['hamming_loss'] = float(m.group(2))

        m = re.search(r'标签级Accuracy\(label_accuracy\):([\d.]+)', line)
        if m:
            target['label_accuracy'] = float(m.group(1))

        m = re.search(r'\*\*Micro P/R/F1:([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            target['micro_f1'] = float(m.group(3))

        m = re.search(r'\*\*Macro P/R/F1:([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            target['macro_precision'] = float(m.group(1))
            target['macro_recall'] = float(m.group(2))
            target['macro_f1'] = float(m.group(3))

        m = re.search(r'\*\*Supported Macro P/R/F1:([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            target['supported_macro_precision'] = float(m.group(1))
            target['supported_macro_recall'] = float(m.group(2))
            target['supported_macro_f1'] = float(m.group(3))

        m = re.search(r'\*\*HitRate\(hit_rate\):([\d.]+), EmptyPredictionRate\(empty_prediction_rate\):([\d.]+)', line)
        if m:
            target['hit_rate'] = float(m.group(1))
            target['empty_prediction_rate'] = float(m.group(2))

        m = re.search(
            r'\*\*平均推荐数:([\d.]+), 平均真实数:([\d.]+), 平均数量误差:([\d.]+), '
            r'平均多选数:([\d.]+), 平均漏选数:([\d.]+)',
            line
        )
        if m:
            target['avg_selected_count'] = float(m.group(1))
            target['avg_true_count'] = float(m.group(2))
            target['avg_cardinality_error'] = float(m.group(3))
            target['avg_over_select'] = float(m.group(4))
            target['avg_under_select'] = float(m.group(5))

        m = re.search(r'\*\*勿选率\(false_selection_rate\):([\d.]+), 漏选率\(miss_selection_rate\):([\d.]+)', line)
        if m:
            target['false_selection_rate'] = float(m.group(1))
            target['miss_selection_rate'] = float(m.group(2))

    for line in lines:
        line = line.strip()

        if '========== 测试集评估开始 ==========' in line or '========== 最终测试集指标汇总 ==========' in line:
            commit_episode()
            collecting_final = True
            current_section = None
            continue

        m = re.match(
            r'.*第 (\d+) 次迭代开始\.\.\.(?: curriculum=([^,]+), samples=(\d+), aux_weight=([\d.\-]+))?',
            line
        )
        if m:
            commit_episode()
            collecting_final = False
            pending_ep = {'episode': int(m.group(1)), 'avg_reward': 0.0, 'avg_loss': 0.0}
            if m.group(2) is not None:
                pending_ep['curriculum'] = m.group(2)
                pending_ep['samples'] = int(m.group(3))
                pending_ep['aux_weight'] = float(m.group(4))
            current_section = None
            continue

        m = re.match(
            r'.*第 (\d+) 次迭代训练统计: avg_reward=([\d.\-]+), avg_loss=([\d.\-]+)'
            r'(?:, avg_selected=([\d.\-]+), agent_actions=(\d+), random_actions=(\d+), '
            r'stop_actions=(\d+), optimize_steps=(\d+), target_updates=(\d+))?',
            line
        )
        if m and pending_ep is not None:
            pending_ep['avg_reward'] = float(m.group(2))
            pending_ep['avg_loss'] = float(m.group(3))
            if m.group(4) is not None:
                pending_ep['avg_selected'] = float(m.group(4))
                pending_ep['agent_actions'] = int(m.group(5))
                pending_ep['random_actions'] = int(m.group(6))
                pending_ep['stop_actions'] = int(m.group(7))
                pending_ep['optimize_steps'] = int(m.group(8))
                pending_ep['target_updates'] = int(m.group(9))
                total_actions = pending_ep['agent_actions'] + pending_ep['random_actions'] + pending_ep['stop_actions']
                if total_actions > 0:
                    pending_ep['agent_action_ratio'] = pending_ep['agent_actions'] / total_actions
                    pending_ep['random_action_ratio'] = pending_ep['random_actions'] / total_actions
                    pending_ep['stop_action_ratio'] = pending_ep['stop_actions'] / total_actions
            continue

        if '**测试集-模型自主停止' in line or '**模型自主停止' in line:
            current_section = 'auto'
            continue

        m = re.search(
            r'测试集-自主停止: sample_f1=([\d.]+), exact_match=([\d.]+), '
            r'micro_f1=([\d.]+), macro_f1=([\d.]+)(?:, supported_macro_f1=([\d.]+))?',
            line
        )
        if m:
            final_metrics['auto'].update({
                'f1': float(m.group(1)),
                'sample_f1': float(m.group(1)),
                'exact_match': float(m.group(2)),
                'micro_f1': float(m.group(3)),
                'macro_f1': float(m.group(4)),
            })
            if m.group(5) is not None:
                final_metrics['auto']['supported_macro_f1'] = float(m.group(5))
            continue

        if current_section is None:
            continue

        if collecting_final:
            target = final_metrics[current_section]
        elif pending_ep is not None:
            target = auto_metrics
        else:
            continue
        parse_metric_line(line, target)

    commit_episode()
    return episodes, final_metrics


def add_table_page(title, rows):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    if not rows:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        return fig

    table = ax.table(
        cellText=rows,
        colLabels=['Metric', 'Auto Stop', 'Note'],
        loc='center',
        cellLoc='center',
        colLoc='center',
        colWidths=[0.30, 0.22, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e9edf5')
        elif row % 2 == 0:
            cell.set_facecolor('#f8f9fb')
    return fig


def format_metric(metrics, key):
    if key not in metrics:
        return '-'
    return f"{metrics[key]:.4f}"


def moving_average(values, window=SMOOTH_WINDOW):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    if window == 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def rolling_std(values, window=SMOOTH_WINDOW):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    result = []
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        result.append(float(np.std(values[start:idx + 1])))
    return np.asarray(result, dtype=float)


def episode_values(episodes, key, default=0.0):
    return [float(item.get(key, default)) for item in episodes]


def shade_curriculum(ax, episodes):
    if not episodes or not any('curriculum' in item for item in episodes):
        return
    current_stage = episodes[0].get('curriculum')
    start_ep = episodes[0]['episode']
    spans = []
    for idx, item in enumerate(episodes[1:], start=1):
        stage = item.get('curriculum')
        if stage != current_stage:
            spans.append((current_stage, start_ep, episodes[idx - 1]['episode']))
            current_stage = stage
            start_ep = item['episode']
    spans.append((current_stage, start_ep, episodes[-1]['episode']))

    used_labels = set()
    for stage, start_ep, end_ep in spans:
        if stage is None:
            continue
        label = f'curriculum={stage}' if stage not in used_labels else None
        used_labels.add(stage)
        ax.axvspan(
            start_ep, end_ep,
            color=CURRICULUM_COLORS.get(stage, '#BDBDBD'),
            alpha=0.08,
            label=label,
        )


def plot_raw_and_smooth(ax, eps, values, title, ylabel, color, smooth_label='Moving Avg'):
    ax.plot(eps, values, color=color, alpha=0.28, linewidth=1.0, label='Raw')
    if len(values) >= 2:
        ax.plot(eps, moving_average(values), color=color, linewidth=2.0, label=smooth_label)
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def is_after_log(log_path):
    try:
        text = open(log_path, 'r', encoding='utf-8').read(12000)
    except OSError:
        return False
    return 'tail_cost_curiosity' in text or "'mode': 'tail_cost_curiosity'" in text


def build_comparison_page(comparison_items):
    comparison_items = [(label, metrics) for label, metrics in comparison_items if metrics]
    if len(comparison_items) < 2:
        return None

    labels = [label for label, _ in comparison_items]
    metric_keys = [key for key, _ in COMPARISON_METRICS]
    metric_labels = [label for _, label in COMPARISON_METRICS]
    x = list(range(len(metric_keys)))
    width = min(0.36, 0.75 / len(comparison_items))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ['#4C78A8', '#D62728', '#59A14F']
    for item_idx, (model_label, metrics) in enumerate(comparison_items):
        offset = (item_idx - (len(comparison_items) - 1) / 2) * width
        values = [metrics.get(key, metrics.get('sample_f1' if key == 'f1' else key, 0.0)) for key in metric_keys]
        bars = ax.bar([pos + offset for pos in x], values, width=width, label=model_label, color=colors[item_idx % len(colors)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.015, 0.98), f'{value:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_title('Before vs After Final Test Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def build_delta_table(comparison_items):
    comparison_items = [(label, metrics) for label, metrics in comparison_items if metrics]
    if len(comparison_items) < 2:
        return None
    before_label, before = comparison_items[0]
    after_label, after = comparison_items[1]
    rows = []
    for key, label in COMPARISON_METRICS:
        before_value = before.get(key, before.get('sample_f1' if key == 'f1' else key))
        after_value = after.get(key, after.get('sample_f1' if key == 'f1' else key))
        if before_value is None or after_value is None:
            rows.append([label, format_metric(before, key), format_metric(after, key), '-'])
        else:
            rows.append([label, f'{before_value:.4f}', f'{after_value:.4f}', f'{after_value - before_value:+.4f}'])
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.set_title('Before vs After Delta Table', fontsize=14, fontweight='bold', pad=16)
    table = ax.table(
        cellText=rows,
        colLabels=['Metric', before_label, after_label, 'Delta'],
        loc='center',
        cellLoc='center',
        colLoc='center',
        colWidths=[0.32, 0.20, 0.20, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e9edf5')
        elif row % 2 == 0:
            cell.set_facecolor('#f8f9fb')
        if row > 0 and col == 3:
            text = cell.get_text().get_text()
            if text.startswith('+'):
                cell.set_facecolor('#e6f4ea')
            elif text.startswith('-'):
                cell.set_facecolor('#fdecea')
    return fig


def build_training_convergence_pages(episodes):
    if not episodes:
        return []

    eps = [item['episode'] for item in episodes]
    rewards = episode_values(episodes, 'avg_reward')
    losses = episode_values(episodes, 'avg_loss')
    figures = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    plot_raw_and_smooth(axes[0], eps, rewards, 'Training Reward Convergence', 'Avg Reward', '#4C78A8')
    plot_raw_and_smooth(axes[1], eps, losses, 'Training Loss Convergence', 'Avg Loss', '#D62728')
    if any('avg_selected' in item for item in episodes):
        selected = episode_values(episodes, 'avg_selected')
        plot_raw_and_smooth(axes[2], eps, selected, 'Avg Selected Count During Training', 'Avg Selected', '#59A14F')
    else:
        axes[2].axis('off')
    if any('aux_weight' in item for item in episodes):
        aux_values = episode_values(episodes, 'aux_weight')
        axes[3].plot(eps, aux_values, color='#F58518', linewidth=2.0)
        axes[3].set_title('Auxiliary Supervised Weight Schedule')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Aux Weight')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].axis('off')
    for ax in axes:
        shade_curriculum(ax, episodes)
    fig.suptitle('Training Convergence Curves', fontsize=14, fontweight='bold')
    fig.tight_layout()
    figures.append(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    reward_std = rolling_std(rewards)
    loss_std = rolling_std(losses)
    axes[0].plot(eps, reward_std, color='#4C78A8', linewidth=2.0)
    axes[0].set_title(f'Reward Rolling Std (window={SMOOTH_WINDOW})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Std')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(eps, loss_std, color='#D62728', linewidth=2.0)
    axes[1].set_title(f'Loss Rolling Std (window={SMOOTH_WINDOW})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Std')
    axes[1].grid(True, alpha=0.3)

    if any('random_action_ratio' in item for item in episodes):
        axes[2].plot(eps, episode_values(episodes, 'agent_action_ratio'), label='Agent', color='#4C78A8')
        axes[2].plot(eps, episode_values(episodes, 'random_action_ratio'), label='Random', color='#D62728')
        axes[2].plot(eps, episode_values(episodes, 'stop_action_ratio'), label='Stop', color='#59A14F')
        axes[2].set_title('Action Source / Stop Ratios')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Ratio')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8)
    else:
        axes[2].axis('off')

    if any('target_updates' in item for item in episodes):
        axes[3].plot(eps, episode_values(episodes, 'target_updates'), color='#9467BD', linewidth=1.8)
        axes[3].set_title('Target Network Updates per Episode')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Updates')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].axis('off')
    for ax in axes:
        shade_curriculum(ax, episodes)
    fig.suptitle('Training Stability Diagnostics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    figures.append(fig)

    return figures


def convergence_summary_rows(episodes):
    if not episodes:
        return []
    rewards = episode_values(episodes, 'avg_reward')
    losses = episode_values(episodes, 'avg_loss')
    tail_count = max(1, min(SMOOTH_WINDOW, len(episodes)))
    first_count = tail_count
    first_reward = float(np.mean(rewards[:first_count]))
    last_reward = float(np.mean(rewards[-tail_count:]))
    first_loss = float(np.mean(losses[:first_count]))
    last_loss = float(np.mean(losses[-tail_count:]))
    reward_tail_std = float(np.std(rewards[-tail_count:]))
    loss_tail_std = float(np.std(losses[-tail_count:]))
    rows = [
        [f'Reward MA first {first_count}', f'{first_reward:.4f}', '训练初期平均reward'],
        [f'Reward MA last {tail_count}', f'{last_reward:.4f}', '训练末期平均reward，越稳定越好'],
        ['Reward MA delta', f'{last_reward - first_reward:+.4f}', '末期-初期，观察是否收敛提升'],
        [f'Reward tail std {tail_count}', f'{reward_tail_std:.4f}', '末期reward波动，越小越平稳'],
        [f'Loss MA first {first_count}', f'{first_loss:.6f}', '训练初期平均loss'],
        [f'Loss MA last {tail_count}', f'{last_loss:.6f}', '训练末期平均loss，通常越低越好'],
        ['Loss MA delta', f'{last_loss - first_loss:+.6f}', '末期-初期，观察优化趋势'],
        [f'Loss tail std {tail_count}', f'{loss_tail_std:.6f}', '末期loss波动，越小越平稳'],
    ]
    if any('avg_selected' in item for item in episodes):
        rows.append(['Last Avg Selected', f"{episodes[-1].get('avg_selected', 0.0):.4f}", '训练末轮平均选择数量'])
    if any('random_action_ratio' in item for item in episodes):
        rows.append(['Last Random Ratio', f"{episodes[-1].get('random_action_ratio', 0.0):.4f}", '训练末轮随机探索占比，应随训练降低'])
    return rows


def save_figures(figures, save_path):
    if not save_path:
        plt.show()
        return

    root, ext = os.path.splitext(save_path)
    ext = ext.lower()
    if ext == '.pdf':
        with PdfPages(save_path) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
        print(f"多页PDF已保存至: {save_path}")
    else:
        for idx, fig in enumerate(figures, start=1):
            page_path = f"{root}_page{idx}{ext or '.png'}"
            fig.savefig(page_path, dpi=150, bbox_inches='tight')
            print(f"图片第{idx}页已保存至: {page_path}")
    plt.close('all')


def plot_metrics(episodes, final_metrics=None, save_path=None, comparison_items=None):
    """核心五个指标用图展示，辅助指标用表格展示。"""
    final_metrics = final_metrics or {'auto': {}}
    if not episodes and not final_metrics.get('auto'):
        print("未找到有效的训练或测试指标数据")
        return

    figures = []
    figures.extend(build_training_convergence_pages(episodes))
    if comparison_items:
        comparison_fig = build_comparison_page(comparison_items)
        if comparison_fig is not None:
            figures.append(comparison_fig)
        delta_fig = build_delta_table(comparison_items)
        if delta_fig is not None:
            figures.append(delta_fig)

    has_episode_eval = any(e.get('auto') for e in episodes)
    if has_episode_eval:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        eps = [e['episode'] for e in episodes]
        for idx, (key, label) in enumerate(CORE_METRICS):
            ax = axes[idx]
            auto_values = [e.get('auto', {}).get(key, e.get('auto', {}).get('sample_f1' if key == 'f1' else key, 0)) for e in episodes]
            ax.plot(eps, auto_values, 'g-o', markersize=4, label='Auto Stop')
            ax.set_title(label)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].axis('off')
        fig.suptitle('Core Metrics Across Episodes', fontsize=14, fontweight='bold')
        fig.tight_layout()
        figures.append(fig)

    auto = final_metrics.get('auto', {})
    if auto:
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_names = [key for key, _ in CORE_METRICS]
        labels = [label for _, label in CORE_METRICS]
        x = list(range(len(metric_names)))
        auto_values = [auto.get(name, 0.0) for name in metric_names]
        ax.bar(x, auto_values, width=0.55, label='Auto Stop')
        ax.set_title('Final Test Core Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        figures.append(fig)

    training_rows = []
    if episodes:
        best_reward = max(episodes, key=lambda item: item['avg_reward'])
        min_loss = min(episodes, key=lambda item: item['avg_loss'])
        training_rows.extend([
            ['Episodes', str(len(episodes)), '训练轮数'],
            ['Last Reward', f"{episodes[-1]['avg_reward']:.4f}", '最后一轮平均奖励'],
            ['Best Reward', f"{best_reward['avg_reward']:.4f}", f"Episode {best_reward['episode']}"],
            ['Last Loss', f"{episodes[-1]['avg_loss']:.6f}", '最后一轮平均损失'],
            ['Min Loss', f"{min_loss['avg_loss']:.6f}", f"Episode {min_loss['episode']}"],
        ])
        training_rows.extend(convergence_summary_rows(episodes))
        figures.append(add_table_page('Training Summary and Convergence Table', training_rows))

    if auto:
        aux_rows = []
        notes = {
            'jaccard': '集合重合度，越高越好',
            'label_accuracy': '标签位准确率，稀疏标签下仅辅助参考',
            'hamming_loss': '标签位错误率，越低越好',
            'micro_f1': '高频标签整体F1，越高越好',
            'hit_rate': '至少命中一个真实标签比例，越高越好',
            'empty_prediction_rate': '空预测比例，通常越低越好',
            'avg_selected_count': '平均推荐数量',
            'avg_true_count': '平均真实数量',
            'avg_cardinality_error': '推荐数量误差，越低越好',
            'avg_over_select': '平均多选数，越低越好',
            'avg_under_select': '平均漏选数，越低越好',
            'false_selection_rate': '样本级FP/(TP+FP)，即勿选率，越低越好',
            'miss_selection_rate': '样本级FN/(TP+FN)，即漏选率，越低越好',
        }
        for key, label in AUX_METRICS:
            aux_rows.append([label, format_metric(auto, key), notes.get(key, '')])
        figures.append(add_table_page('Auxiliary Final Test Metrics Table', aux_rows))

    save_figures(figures, save_path)


def find_latest_log(after=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    search_dirs = [
        script_dir,
        os.getcwd(),
        os.path.dirname(script_dir),
        os.path.join(os.path.dirname(script_dir), 'ablations', 'results'),
        os.path.join(project_root, 'syj', 'lhz', 'ablations', 'results'),
    ]
    log_files = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            log_files.extend(glob.glob(os.path.join(search_dir, 'dqn_*.log')))
            log_files.extend(glob.glob(os.path.join(search_dir, '**', 'dqn_*.log'), recursive=True))
    existing = [path for path in set(log_files) if os.path.exists(path)]
    if after is not None:
        existing = [path for path in existing if is_after_log(path) == after]
    return max(existing, key=os.path.getmtime) if existing else None


def parse_args():
    parser = argparse.ArgumentParser(description='解析DQN训练日志并绘制训练/评估指标。')
    parser.add_argument('log_path', nargs='?', help='主日志路径；不传则优先使用最新改动后日志')
    parser.add_argument('--before-log', default=None, help='改动前模型日志；默认自动查找最新非tail_cost_curiosity日志')
    parser.add_argument('--after-log', default=None, help='改动后模型日志；默认自动查找最新tail_cost_curiosity日志')
    parser.add_argument('--output', default=None, help='输出PDF/图片路径；默认基于主日志名生成')
    parser.add_argument('--no-compare', action='store_true', help='不生成改动前后对比页')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    before_log = args.before_log or find_latest_log(after=False)
    after_log = args.after_log or find_latest_log(after=True)
    log_path = args.log_path or after_log or before_log or find_latest_log()
    if log_path is None:
        print("未找到日志文件，请指定日志文件路径: python plot_training.py <log_path>")
        print("也可以先运行: python syj/lhz/mymodel/main.py 或 python syj/lhz/mymodel_reward_v2/main.py")
        sys.exit(1)
    print(f"主日志: {log_path}")

    episodes, final_metrics = parse_log(log_path)
    print(f"解析到 {len(episodes)} 个 episode 的训练数据")
    if final_metrics.get('auto'):
        print("解析到最终测试集指标")
    if episodes:
        for ep_data in episodes[:3] + ([] if len(episodes) <= 3 else [episodes[-1]]):
            print(f"  Episode {ep_data['episode']}: "
                  f"reward={ep_data['avg_reward']:.4f}, loss={ep_data['avg_loss']:.6f}, "
                  f"auto_keys={list(ep_data.get('auto', {}).keys())}")
    comparison_items = []
    if not args.no_compare:
        if before_log:
            _, before_metrics = parse_log(before_log)
            comparison_items.append(('Before', before_metrics.get('auto', {})))
            print(f"改动前日志: {before_log}")
        if after_log:
            _, after_metrics = parse_log(after_log)
            comparison_items.append(('After', after_metrics.get('auto', {})))
            print(f"改动后日志: {after_log}")
    save_path = args.output or log_path.replace('.log', '_metrics.pdf')
    plot_metrics(episodes, final_metrics, save_path, comparison_items=comparison_items)
