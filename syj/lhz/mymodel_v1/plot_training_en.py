# -*- coding: utf-8 -*-
"""Plot structured V1 training history without parsing legacy logs."""

import argparse
import json
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent / 'runs'


def find_latest_run(runs_dir=RUNS_DIR):
    """Select the most recently updated run containing a non-empty history.json."""
    candidates = []
    for path in Path(runs_dir).rglob('history.json'):
        try:
            if json.loads(path.read_text(encoding='utf-8')):
                candidates.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(f'No valid history.json found under {runs_dir}')
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def load_history(run_dir):
    path = Path(run_dir) / 'history.json'
    if not path.exists():
        raise FileNotFoundError(f'Training history not found: {path}')
    history = json.loads(path.read_text(encoding='utf-8'))
    if not history:
        raise ValueError('history.json is empty; this run may not have completed an RL epoch')
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
    except ImportError as exc:
        raise RuntimeError('matplotlib is required for plotting; please install it first') from exc

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
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        if early_stopped and stop_epoch != best_epoch:
            axis.axvline(stop_epoch, color='#C44E52', linestyle=':', linewidth=1.5,
                         label=f'Early stop epoch {stop_epoch}')

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    axes = axes.ravel()

    axes[0].plot(epochs, rewards, marker='o', markersize=3, color='#4C78A8')
    label_axis(axes[0], 'Average Training Reward', 'Reward')

    axes[1].plot(epochs, losses, marker='o', markersize=3, color='#F28E2B')
    if all(value > 0 for value in losses):
        axes[1].set_yscale('log')
    label_axis(axes[1], 'TD Loss (Log Scale)', 'Loss')

    f1_metrics = [
        ('sample_f1', 'Sample F1'),
        ('micro_f1', 'Micro F1'),
        ('supported_macro_f1', 'Supported Macro-F1'),
    ]
    for key, label in f1_metrics:
        axes[2].plot(epochs, values(key), marker='o', markersize=2.5, label=label)
    label_axis(axes[2], 'Validation F1 Metrics', 'F1')
    axes[2].legend(fontsize=8)

    quality_metrics = [
        ('sample_precision', 'Sample Precision'),
        ('sample_recall', 'Sample Recall'),
        ('exact_match', 'Exact Match'),
    ]
    for key, label in quality_metrics:
        axes[3].plot(epochs, values(key), marker='o', markersize=2.5, label=label)
    label_axis(axes[3], 'Validation Precision, Recall, and Exact Match', 'Rate')
    axes[3].legend(fontsize=8)

    selection_metrics = [
        ('miss_selection_rate', 'Missed Selection Rate'),
        ('false_selection_rate', 'False Selection Rate'),
    ]
    for key, label in selection_metrics:
        axes[4].plot(epochs, values(key), marker='o', markersize=2.5, label=label)
    label_axis(axes[4], 'Validation Selection Diagnostics', 'Rate')
    axes[4].legend(fontsize=8)

    axes[5].plot(epochs, values('avg_selected_count'), marker='o', markersize=2.5,
                 label='Validation Avg. Predicted Labels')
    axes[5].plot(epochs, values('avg_true_count'), linestyle='--',
                 label='Validation Avg. Ground-truth Labels')
    axes[5].plot(epochs, [row.get('avg_selected', 0.0) for row in history], marker='o',
                 markersize=2.5, label='Training Avg. Predicted Labels')
    label_axis(axes[5], 'Training and Validation Label Counts', 'Average Label Count')
    axes[5].legend(fontsize=8)

    status = f'Early stop at epoch {stop_epoch}' if early_stopped else f'Completed {len(history)} epochs'
    figure.suptitle(f'Training and Validation Curves: Best Epoch = {best_epoch}, {status}', fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return best_epoch, stop_epoch, early_stopped


def main():
    parser = argparse.ArgumentParser(description='Plot mymodel_v1 training curves in English')
    parser.add_argument('--run-dir', default=None,
                        help='V1 run directory containing history.json; defaults to the latest run')
    parser.add_argument('--output', default=None,
                        help='Output PNG path; defaults to the run directory')
    parser.add_argument('--metric', default='supported_macro_f1',
                        help='Validation metric name in history.json')
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    output = Path(args.output) if args.output else run_dir / 'training_curves_en.png'
    history = load_history(run_dir)
    run_config = load_run_config(run_dir)
    best_epoch, stop_epoch, early_stopped = plot_history(history, output, args.metric, run_config)
    print(f'Training directory: {run_dir}')
    print(f'Actual training: {len(history)} epochs (epoch {history[0]["epoch"]}-{stop_epoch})')
    print(f'Best validation result: epoch {best_epoch}, selection metric = {args.metric}')
    print(f'Training completion: {"validation early stopping" if early_stopped else "configured epochs completed"}')
    print(f'Written to: {output.resolve()}')


if __name__ == '__main__':
    main()
