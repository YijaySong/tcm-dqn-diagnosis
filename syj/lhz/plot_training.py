# -*- coding: utf-8 -*-
"""解析训练日志，绘制训练曲线和最终测试集指标"""
import os
import glob
import re
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


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
        m = re.search(r'\*\*样本Jaccard avg:([\d.]+)', line)
        if m:
            target['jaccard'] = float(m.group(1))

        m = re.search(r'\*\*样本Precision avg:([\d.]+), Recall avg:([\d.]+), F1 avg:([\d.]+)', line)
        if m:
            target['precision'] = float(m.group(1))
            target['recall'] = float(m.group(2))
            target['f1'] = float(m.group(3))
            target['sample_f1'] = float(m.group(3))

        m = re.search(r'\*\*ExactMatch:([\d.]+), HammingLoss:([\d.]+)', line)
        if m:
            target['exact_match'] = float(m.group(1))
            target['hamming_loss'] = float(m.group(2))

        m = re.search(r'\*\*Micro P/R/F1:([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            target['micro_f1'] = float(m.group(3))

        m = re.search(r'\*\*Macro P/R/F1:([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            target['macro_f1'] = float(m.group(3))

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


def plot_metrics(episodes, final_metrics=None, save_path=None):
    """绘制训练指标曲线和最终测试集指标。"""
    final_metrics = final_metrics or {'auto': {}, 'top2': {}}
    if not episodes and not final_metrics.get('auto') and not final_metrics.get('top2'):
        print("未找到有效的训练或测试指标数据")
        return

    _, axes = plt.subplots(2, 3, figsize=(16, 10))

    if episodes:
        eps = [e['episode'] for e in episodes]
        rewards = [e['avg_reward'] for e in episodes]
        losses = [e['avg_loss'] for e in episodes]

        axes[0, 0].plot(eps, rewards, 'b-o', markersize=4)
        axes[0, 0].set_title('Avg Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Avg Reward')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(eps, losses, 'r-o', markersize=4)
        axes[0, 1].set_title('Avg Loss per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Avg Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')

    has_episode_eval = any(e.get('auto') or e.get('top2') for e in episodes)
    if has_episode_eval:
        eps = [e['episode'] for e in episodes]
        auto_f1 = [e.get('auto', {}).get('f1', 0) for e in episodes]
        auto_em = [e.get('auto', {}).get('exact_match', 0) for e in episodes]
        auto_micro_f1 = [e.get('auto', {}).get('micro_f1', 0) for e in episodes]
        auto_macro_f1 = [e.get('auto', {}).get('macro_f1', 0) for e in episodes]
        top2_f1 = [e.get('top2', {}).get('f1', 0) for e in episodes]
        top2_em = [e.get('top2', {}).get('exact_match', 0) for e in episodes]

        axes[0, 2].plot(eps, auto_f1, 'g-o', markersize=4, label='Auto Stop')
        axes[0, 2].plot(eps, top2_f1, 'orange', marker='s', markersize=4, label='Top-2')
        axes[0, 2].set_title('Episode Sample F1')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('F1')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(eps, auto_em, 'g-o', markersize=4, label='Auto Stop')
        axes[1, 0].plot(eps, top2_em, 'orange', marker='s', markersize=4, label='Top-2')
        axes[1, 0].set_title('Episode Exact Match')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Exact Match')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(eps, auto_micro_f1, 'c-o', markersize=4, label='Micro F1')
        axes[1, 1].plot(eps, auto_macro_f1, 'm-s', markersize=4, label='Macro F1')
        axes[1, 1].set_title('Auto Stop: Micro / Macro F1')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('F1')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[0, 2].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    auto = final_metrics.get('auto', {})
    top2 = final_metrics.get('top2', {})
    if auto or top2:
        metric_names = ['f1', 'exact_match', 'micro_f1', 'macro_f1']
        labels = ['Sample F1', 'Exact Match', 'Micro F1', 'Macro F1']
        x = range(len(metric_names))
        width = 0.35
        auto_values = [auto.get(name, 0.0) for name in metric_names]
        top2_values = [top2.get(name, 0.0) for name in metric_names]

        axes[1, 1].clear()
        axes[1, 1].bar([i - width / 2 for i in x], auto_values, width, label='Auto Stop')
        axes[1, 1].bar([i + width / 2 for i in x], top2_values, width, label='Top-2')
        axes[1, 1].set_title('Final Test Metrics')
        axes[1, 1].set_xticks(list(x))
        axes[1, 1].set_xticklabels(labels, rotation=20)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis='y', alpha=0.3)

    axes[1, 2].axis('off')
    text_lines = []
    if episodes:
        text_lines.append(f"Episodes: {len(episodes)}")
        text_lines.append(f"Last Reward: {episodes[-1]['avg_reward']:.4f}")
        text_lines.append(f"Last Loss: {episodes[-1]['avg_loss']:.6f}")
    if auto:
        text_lines.extend([
            "",
            "Final Test - Auto Stop",
            f"Sample F1: {auto.get('f1', 0):.4f}",
            f"Exact Match: {auto.get('exact_match', 0):.4f}",
            f"Micro F1: {auto.get('micro_f1', 0):.4f}",
            f"Macro F1: {auto.get('macro_f1', 0):.4f}",
        ])
    if top2:
        text_lines.extend([
            "",
            "Final Test - Top-2",
            f"Sample F1: {top2.get('f1', 0):.4f}",
            f"Exact Match: {top2.get('exact_match', 0):.4f}",
            f"Micro F1: {top2.get('micro_f1', 0):.4f}",
            f"Macro F1: {top2.get('macro_f1', 0):.4f}",
        ])
    axes[1, 2].text(0.05, 0.95, '\n'.join(text_lines), fontsize=11, family='monospace',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('DQN Training and Test Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.show()


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
    save_path = log_path.replace('.log', '_metrics.png')
    plot_metrics(episodes, final_metrics, save_path)
