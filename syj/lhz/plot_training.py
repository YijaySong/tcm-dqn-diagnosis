# -*- coding: utf-8 -*-
"""训练日志绘图文件。

负责解析DQN训练过程中生成的日志文件，提取reward、loss和测试集评估指标。
输出时只把最重要的五个模型指标（sample_f1、sample_recall、sample_precision、
exact_match、macro_f1）画成图；其他辅助指标和训练摘要用表格展示。默认保存为
多页PDF，便于分开展示核心图表和辅助信息。
"""
import os
import glob
import re
import sys

import matplotlib
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
]


def parse_log(log_path):
    """解析日志文件。

    兼容两种日志：
    1. 旧日志：每个 episode 后都有模型自主停止/Top-2评估。
    2. 新日志：episode 内只记录训练 reward/loss，训练完成后单独记录测试集指标。
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    episodes = []
    final_metrics = {'auto': {}, 'top2': {}}
    pending_ep = None
    current_section = None
    auto_metrics = {}
    top2_metrics = {}
    collecting_final = False

    def commit_episode():
        nonlocal pending_ep, auto_metrics, top2_metrics
        if pending_ep is not None:
            pending_ep['auto'] = auto_metrics.copy()
            pending_ep['top2'] = top2_metrics.copy()
            episodes.append(pending_ep)
        pending_ep = None
        auto_metrics = {}
        top2_metrics = {}

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
            target['macro_f1'] = float(m.group(3))

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

    for line in lines:
        line = line.strip()

        if '========== 测试集评估开始 ==========' in line or '========== 最终测试集指标汇总 ==========' in line:
            commit_episode()
            collecting_final = True
            current_section = None
            continue

        m = re.match(r'.*第 (\d+) 次迭代开始\.\.\.', line)
        if m:
            commit_episode()
            collecting_final = False
            pending_ep = {'episode': int(m.group(1)), 'avg_reward': 0.0, 'avg_loss': 0.0}
            current_section = None
            continue

        m = re.match(r'.*第 (\d+) 次迭代训练统计: avg_reward=([\d.\-]+), avg_loss=([\d.\-]+)', line)
        if m and pending_ep is not None:
            pending_ep['avg_reward'] = float(m.group(2))
            pending_ep['avg_loss'] = float(m.group(3))
            continue

        if '**测试集-模型自主停止' in line or '**模型自主停止' in line:
            current_section = 'auto'
            continue
        if '**测试集-固定Top-2诊断' in line or '**固定Top-2诊断' in line:
            current_section = 'top2'
            continue

        m = re.search(r'测试集-自主停止: sample_f1=([\d.]+), exact_match=([\d.]+), micro_f1=([\d.]+), macro_f1=([\d.]+)', line)
        if m:
            final_metrics['auto'].update({
                'f1': float(m.group(1)),
                'sample_f1': float(m.group(1)),
                'exact_match': float(m.group(2)),
                'micro_f1': float(m.group(3)),
                'macro_f1': float(m.group(4)),
            })
            continue

        m = re.search(r'测试集-Top-2: sample_f1=([\d.]+), exact_match=([\d.]+), micro_f1=([\d.]+), macro_f1=([\d.]+)', line)
        if m:
            final_metrics['top2'].update({
                'f1': float(m.group(1)),
                'sample_f1': float(m.group(1)),
                'exact_match': float(m.group(2)),
                'micro_f1': float(m.group(3)),
                'macro_f1': float(m.group(4)),
            })
            continue

        if current_section is None:
            continue

        if collecting_final:
            target = final_metrics[current_section]
        elif pending_ep is not None:
            target = auto_metrics if current_section == 'auto' else top2_metrics
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
        colLabels=['Metric', 'Auto Stop', 'Top-2', 'Note'],
        loc='center',
        cellLoc='center',
        colLoc='center',
        colWidths=[0.25, 0.18, 0.18, 0.32],
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
    plt.show()


def plot_metrics(episodes, final_metrics=None, save_path=None):
    """核心五个指标用图展示，辅助指标用表格展示。"""
    final_metrics = final_metrics or {'auto': {}, 'top2': {}}
    if not episodes and not final_metrics.get('auto') and not final_metrics.get('top2'):
        print("未找到有效的训练或测试指标数据")
        return

    figures = []
    has_episode_eval = any(e.get('auto') or e.get('top2') for e in episodes)
    if has_episode_eval:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        eps = [e['episode'] for e in episodes]
        for idx, (key, label) in enumerate(CORE_METRICS):
            ax = axes[idx]
            auto_values = [e.get('auto', {}).get(key, e.get('auto', {}).get('sample_f1' if key == 'f1' else key, 0)) for e in episodes]
            top2_values = [e.get('top2', {}).get(key, e.get('top2', {}).get('sample_f1' if key == 'f1' else key, 0)) for e in episodes]
            ax.plot(eps, auto_values, 'g-o', markersize=4, label='Auto Stop')
            ax.plot(eps, top2_values, color='orange', marker='s', markersize=4, label='Top-2')
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
    top2 = final_metrics.get('top2', {})
    if auto or top2:
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_names = [key for key, _ in CORE_METRICS]
        labels = [label for _, label in CORE_METRICS]
        x = list(range(len(metric_names)))
        width = 0.35
        auto_values = [auto.get(name, 0.0) for name in metric_names]
        top2_values = [top2.get(name, 0.0) for name in metric_names]
        ax.bar([i - width / 2 for i in x], auto_values, width, label='Auto Stop')
        ax.bar([i + width / 2 for i in x], top2_values, width, label='Top-2')
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
            ['Episodes', str(len(episodes)), '-', '训练轮数'],
            ['Last Reward', f"{episodes[-1]['avg_reward']:.4f}", '-', '最后一轮平均奖励'],
            ['Best Reward', f"{best_reward['avg_reward']:.4f}", '-', f"Episode {best_reward['episode']}"],
            ['Last Loss', f"{episodes[-1]['avg_loss']:.6f}", '-', '最后一轮平均损失'],
            ['Min Loss', f"{min_loss['avg_loss']:.6f}", '-', f"Episode {min_loss['episode']}"],
        ])
        figures.append(add_table_page('Training Summary Table', training_rows))

    if auto or top2:
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
        }
        for key, label in AUX_METRICS:
            aux_rows.append([label, format_metric(auto, key), format_metric(top2, key), notes.get(key, '')])
        figures.append(add_table_page('Auxiliary Final Test Metrics Table', aux_rows))

    save_figures(figures, save_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_files = glob.glob(os.path.join(script_dir, '..', '..', 'dqn_*.log'))
        if not log_files:
            log_files = glob.glob(os.path.join(script_dir, 'dqn_*.log'))
        if not log_files:
            print("未找到日志文件，请指定日志文件路径: python plot_training.py <log_path>")
            sys.exit(1)
        log_path = max(log_files, key=os.path.getmtime)
        print(f"自动选择最新日志: {log_path}")
    else:
        log_path = sys.argv[1]

    episodes, final_metrics = parse_log(log_path)
    print(f"解析到 {len(episodes)} 个 episode 的训练数据")
    if final_metrics.get('auto') or final_metrics.get('top2'):
        print("解析到最终测试集指标")
    if episodes:
        for ep_data in episodes[:3] + ([] if len(episodes) <= 3 else [episodes[-1]]):
            print(f"  Episode {ep_data['episode']}: "
                  f"reward={ep_data['avg_reward']:.4f}, loss={ep_data['avg_loss']:.6f}, "
                  f"auto_keys={list(ep_data.get('auto', {}).keys())}, top2_keys={list(ep_data.get('top2', {}).keys())}")
    save_path = log_path.replace('.log', '_metrics.pdf')
    plot_metrics(episodes, final_metrics, save_path)
