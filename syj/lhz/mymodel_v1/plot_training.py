# -*- coding: utf-8 -*-
"""绘制 V1 的结构化训练历史，不解析旧版日志。"""

import argparse
import json
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent / 'runs'


def find_latest_run(runs_dir=RUNS_DIR):
    """选择最近写入且包含非空 history.json 的训练目录。"""
    candidates = []
    for path in Path(runs_dir).rglob('history.json'):
        try:
            if json.loads(path.read_text(encoding='utf-8')):
                candidates.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(f'{runs_dir} 下没有包含有效 history.json 的训练结果')
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def load_history(run_dir):
    path = Path(run_dir) / 'history.json'
    if not path.exists():
        raise FileNotFoundError(f'未找到训练历史: {path}')
    history = json.loads(path.read_text(encoding='utf-8'))
    if not history:
        raise ValueError('history.json 为空；该 run 可能未执行 RL epoch')
    return history


def load_run_config(run_dir):
    path = Path(run_dir) / 'resolved_config.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def plot_history(history, output_path, metric, run_config=None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:
        raise RuntimeError('绘图需要 matplotlib，请先安装后重试') from exc

    preferred_font_files = [
        Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
        Path('/Library/Fonts/Arial Unicode.ttf'),
        Path('C:/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/simhei.ttf'),
    ]
    font_path = next((path for path in preferred_font_files if path.exists()), None)
    cjk_font = FontProperties(fname=str(font_path)) if font_path else None

    epochs = [row['epoch'] for row in history]
    rewards = [row.get('avg_reward', 0.0) for row in history]
    losses = [row.get('avg_loss', 0.0) for row in history]
    validation_rows = [row.get('validation', {}) for row in history]
    improved_rows = [row for row in history if row.get('improved')]
    best_row = improved_rows[-1] if improved_rows else max(
        history, key=lambda row: row.get('validation', {}).get(metric, -float('inf'))
    )
    best_epoch = best_row['epoch']
    stop_epoch = epochs[-1]
    run_config = run_config or {}
    configured_episodes = int(run_config.get('episodes', len(history)))
    patience = int(run_config.get('patience', 0))
    early_stopped = patience > 0 and len(history) < configured_episodes

    def values(key, default=0.0):
        return [row.get(key, default) for row in validation_rows]

    def label_axis(axis, title, ylabel):
        axis.set_title(title, fontproperties=cjk_font)
        axis.set_xlabel('Epoch')
        axis.set_ylabel(ylabel, fontproperties=cjk_font)
        axis.grid(alpha=0.2)
        if early_stopped and stop_epoch != best_epoch:
            axis.axvline(stop_epoch, color='#C44E52', linestyle=':', linewidth=1.5, label=f'早停 epoch {stop_epoch}')

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    axes = axes.ravel()

    axes[0].plot(epochs, rewards, marker='o', markersize=3, color='#4C78A8')
    label_axis(axes[0], '训练平均奖励', 'Reward')

    axes[1].plot(epochs, losses, marker='o', markersize=3, color='#F28E2B')
    if all(value > 0 for value in losses):
        axes[1].set_yscale('log')
    label_axis(axes[1], 'TD Loss（对数坐标）', 'Loss')

    f1_metrics = [
    ('sample_f1', 'Sample F1'),
    ('micro_f1', 'Micro F1'),
    ('supported_macro_f1', 'Supported Macro-F1'),
    ]
    for key, label in f1_metrics:
        axes[2].plot(epochs, values(key), marker='o', markersize=2.5, label=label)
    label_axis(axes[2], '验证集 F1 指标', 'F1')
    axes[2].legend(prop=cjk_font, fontsize=8)

    quality_metrics = [
        ('sample_precision', 'Sample Precision'),
        ('sample_recall', 'Sample Recall'),
        ('exact_match', 'Exact Match'),
    ]
    for key, label in quality_metrics:
        axes[3].plot(epochs, values(key), marker='o', markersize=2.5, label=label)
    label_axis(axes[3], '验证集精确率、召回率与完全匹配', 'Rate')
    axes[3].legend(prop=cjk_font, fontsize=8)

    stop_metrics = [
        ('miss_selection_rate', '漏选率'),
        ('false_selection_rate', '误选率'),
    ]
    for key, label in stop_metrics:
        axes[4].plot(epochs, values(key), marker='o', markersize=2.5, label=label)
    label_axis(axes[4], '验证集停止与选择诊断', 'Rate')
    axes[4].legend(prop=cjk_font, fontsize=8)

    axes[5].plot(epochs, values('avg_selected_count'), marker='o', markersize=2.5, label='验证集平均预测数')
    axes[5].plot(epochs, values('avg_true_count'), linestyle='--', label='验证集平均真值数')
    axes[5].plot(epochs, [row.get('avg_selected', 0.0) for row in history], marker='o', markersize=2.5, label='训练集平均预测数')
    label_axis(axes[5], '训练集与验证集标签数量', '平均标签数')
    axes[5].legend(prop=cjk_font, fontsize=8)

    status = f'早停 epoch={stop_epoch}' if early_stopped else f'完整训练 {len(history)} 轮'
    figure.suptitle(f'训练与验证曲线：最佳 epoch={best_epoch}，{status}',
                    fontproperties=cjk_font, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return best_epoch, stop_epoch, early_stopped


def main():
    parser = argparse.ArgumentParser(description='绘制 mymodel_v1 的训练曲线')
    parser.add_argument('--run-dir', default=None, help='含 history.json 的 V1 run 目录；默认自动选择最新结果')
    parser.add_argument('--output', default=None, help='PNG 输出路径；默认写入 run 目录')
    parser.add_argument('--metric', default='supported_macro_f1', help='history 中的验证指标名')
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    output = Path(args.output) if args.output else run_dir / 'training_curves.png'
    history = load_history(run_dir)
    run_config = load_run_config(run_dir)
    best_epoch, stop_epoch, early_stopped = plot_history(history, output, args.metric, run_config)
    print(f'训练目录: {run_dir}')
    print(f'实际训练: {len(history)} 轮（epoch {history[0]["epoch"]}-{stop_epoch}）')
    print(f'验证集最佳: epoch {best_epoch}，选模指标={args.metric}')
    print(f'训练结束方式: {"验证早停" if early_stopped else "已跑满设定轮数"}')
    print(f'已写入: {output.resolve()}')


if __name__ == '__main__':
    main()
