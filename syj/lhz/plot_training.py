# -*- coding: utf-8 -*-
"""解析训练日志，绘制训练指标随 episode 变化曲线"""
import re
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_log(log_path):
    """解析日志文件，提取每个 episode 的指标。
    
    日志结构（每个episode）:
        第 X 次迭代开始...
        （训练过程...）
        evaluate...
        **模型自主停止
        （auto指标...）
        **固定Top-2诊断
        （top2指标...）
        第 X 次迭代训练统计: avg_reward=..., avg_loss=...
        
    策略：遇到"迭代开始"时重置状态并暂存上一轮的episode（如果有数据）；
          遇到"训练统计"时记录reward/loss，同时标记当前episode完成。
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    episodes = []
    pending_ep = None  # 当前正在收集的 episode
    current_section = None  # 'auto' or 'top2'
    auto_metrics = {}
    top2_metrics = {}

    def commit_episode():
        """将当前 episode 数据存入列表"""
        nonlocal pending_ep, auto_metrics, top2_metrics
        if pending_ep is not None and (auto_metrics or top2_metrics):
            pending_ep['auto'] = auto_metrics.copy()
            pending_ep['top2'] = top2_metrics.copy()
            episodes.append(pending_ep)
        pending_ep = None
        auto_metrics = {}
        top2_metrics = {}

    for line in lines:
        line = line.strip()

        # 匹配 episode 开始 -> 提交上一轮，开始新一轮
        m = re.match(r'.*第 (\d+) 次迭代开始\.\.\.', line)
        if m:
            commit_episode()
            pending_ep = {'episode': int(m.group(1)), 'avg_reward': 0.0, 'avg_loss': 0.0}
            current_section = None
            continue

        # 如果没有 pending_ep，跳过（还没进入任何 episode）
        if pending_ep is None:
            continue

        # 匹配训练统计行 -> 填充 reward/loss
        m = re.match(r'.*第 (\d+) 次迭代训练统计: avg_reward=([\d.\-]+), avg_loss=([\d.\-]+)', line)
        if m:
            pending_ep['avg_reward'] = float(m.group(2))
            pending_ep['avg_loss'] = float(m.group(3))
            continue

        # 匹配评估段标题
        if '**模型自主停止' in line:
            current_section = 'auto'
            continue
        if '**固定Top-2诊断' in line:
            current_section = 'top2'
            continue

        # 解析指标行
        target = auto_metrics if current_section == 'auto' else (top2_metrics if current_section == 'top2' else None)
        if target is None:
            continue

        # 样本Jaccard
        m = re.search(r'\*\*样本Jaccard avg:([\d.]+)', line)
        if m:
            target['jaccard'] = float(m.group(1))
        # 样本Precision/Recall/F1
        m = re.search(r'\*\*样本Precision avg:([\d.]+), Recall avg:([\d.]+), F1 avg:([\d.]+)', line)
        if m:
            target['precision'] = float(m.group(1))
            target['recall'] = float(m.group(2))
            target['f1'] = float(m.group(3))
        # ExactMatch
        m = re.search(r'\*\*ExactMatch:([\d.]+)', line)
        if m:
            target['exact_match'] = float(m.group(1))
        # Micro P/R/F1
        m = re.search(r'\*\*Micro P/R/F1:([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            target['micro_f1'] = float(m.group(3))
        # Macro P/R/F1
        m = re.search(r'\*\*Macro P/R/F1:([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            target['macro_f1'] = float(m.group(3))

    # 保存最后一个 episode
    commit_episode()

    return episodes


def plot_metrics(episodes, save_path=None):
    """绘制训练指标曲线"""
    if not episodes:
        print("未找到有效的 episode 数据")
        return

    eps = [e['episode'] for e in episodes]
    rewards = [e['avg_reward'] for e in episodes]
    losses = [e['avg_loss'] for e in episodes]
    auto_f1 = [e['auto'].get('f1', 0) for e in episodes]
    auto_em = [e['auto'].get('exact_match', 0) for e in episodes]
    auto_micro_f1 = [e['auto'].get('micro_f1', 0) for e in episodes]
    auto_macro_f1 = [e['auto'].get('macro_f1', 0) for e in episodes]
    top2_f1 = [e['top2'].get('f1', 0) for e in episodes]
    top2_em = [e['top2'].get('exact_match', 0) for e in episodes]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 图1: 平均奖励
    axes[0, 0].plot(eps, rewards, 'b-o', markersize=4)
    axes[0, 0].set_title('Avg Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Avg Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 平均损失
    axes[0, 1].plot(eps, losses, 'r-o', markersize=4)
    axes[0, 1].set_title('Avg Loss per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Avg Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 样本F1（自主停止 vs Top-2）
    axes[0, 2].plot(eps, auto_f1, 'g-o', markersize=4, label='Auto Stop')
    axes[0, 2].plot(eps, top2_f1, 'orange', marker='s', markersize=4, label='Top-2')
    axes[0, 2].set_title('Sample F1')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('F1')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 图4: ExactMatch（自主停止 vs Top-2）
    axes[1, 0].plot(eps, auto_em, 'g-o', markersize=4, label='Auto Stop')
    axes[1, 0].plot(eps, top2_em, 'orange', marker='s', markersize=4, label='Top-2')
    axes[1, 0].set_title('Exact Match')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Exact Match')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 图5: Micro / Macro F1（自主停止）
    axes[1, 1].plot(eps, auto_micro_f1, 'c-o', markersize=4, label='Micro F1')
    axes[1, 1].plot(eps, auto_macro_f1, 'm-s', markersize=4, label='Macro F1')
    axes[1, 1].set_title('Auto Stop: Micro / Macro F1')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('F1')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 图6: 表格汇总
    axes[1, 2].axis('off')
    best_idx = max(range(len(eps)), key=lambda i: auto_f1[i])
    text = (
        f"Best Episode: {eps[best_idx]}\n"
        f"Best Auto-Stop F1: {auto_f1[best_idx]:.4f}\n"
        f"Best Auto-Stop EM: {auto_em[best_idx]:.4f}\n"
        f"Best Auto-Stop Micro F1: {auto_micro_f1[best_idx]:.4f}\n"
        f"Best Auto-Stop Macro F1: {auto_macro_f1[best_idx]:.4f}\n"
        f"Best Top-2 F1: {top2_f1[best_idx]:.4f}\n"
    )
    axes[1, 2].text(0.1, 0.7, text, fontsize=12, family='monospace',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('DQN Training Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # 默认查找同目录下最新的 .log 文件
        import os
        import glob
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

    episodes = parse_log(log_path)
    print(f"解析到 {len(episodes)} 个 episode 的指标数据")
    if episodes:
        # 打印前3个和后1个 episode 的解析结果，方便调试
        for ep_data in episodes[:3] + [episodes[-1]]:
            print(f"  Episode {ep_data['episode']}: "
                  f"reward={ep_data['avg_reward']:.4f}, loss={ep_data['avg_loss']:.6f}, "
                  f"auto_keys={list(ep_data['auto'].keys())}, top2_keys={list(ep_data['top2'].keys())}")
    save_path = log_path.replace('.log', '_metrics.png')
    plot_metrics(episodes, save_path)
