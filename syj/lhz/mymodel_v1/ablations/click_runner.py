# -*- coding: utf-8 -*-
"""供各消融子目录的 run.py 复用的一键运行与绘图工具。"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from common import FAMILIES, ROOT, run_family

sys.path.insert(0, str(ROOT))
from final_config import FINAL_CONFIG, FINAL_CONFIG_VERSION


# 正式消融默认配置。用户只需点击运行，不需要输入命令行参数。
FORMAL_SEEDS = [9, 17, 29]
FORMAL_EPISODES = 50
FORMAL_PRETRAIN_EPOCHS = 10


FINAL_CONFIG_CLI_NAMES = {
    'nn_units': '--nn-units',
    'nn_units2': '--nn-units2',
    'dropout': '--dropout',
    'model_type': '--model-type',
    'exploration_mode': '--exploration-mode',
    'eps_start': '--eps-start',
    'eps_end': '--eps-end',
    'eps_decay': '--eps-decay',
    'min_actions_before_stop': '--min-actions-before-stop',
    'stop_margin_threshold': '--stop-margin-threshold',
    'gamma': '--gamma',
    'reward_beta': '--reward-beta',
    'step_cost': '--step-cost',
    'stop_utility_scale': '--stop-utility-scale',
    'weight_min': '--weight-min',
    'weight_max': '--weight-max',
    'weight_power': '--weight-power',
    'use_pretrain': '--use-pretrain',
    'pretrain_epochs': '--pretrain-epochs',
    'pretrain_lr': '--pretrain-lr',
    'pretrain_num_orders': '--pretrain-num-orders',
    'use_expert_warmup': '--use-expert-warmup',
    'use_aux_bce': '--use-aux-bce',
    'aux_bce_weight': '--aux-bce-weight',
    'run_rl': '--run-rl',
    'supervised_loss': '--supervised-loss',
    'focal_gamma': '--focal-gamma',
    'use_per': '--use-per',
    'rare_per_scale': '--rare-per-scale',
}


def _frozen_final_config_args():
    """将点击运行瞬间的最终配置显式传给每个子进程。"""
    missing = sorted(set(FINAL_CONFIG) - set(FINAL_CONFIG_CLI_NAMES))
    if missing:
        raise RuntimeError(f'以下最终配置尚未加入消融冻结映射: {missing}')
    args = []
    for key, cli_name in FINAL_CONFIG_CLI_NAMES.items():
        value = FINAL_CONFIG[key]
        args.extend([cli_name, 'none' if value is None else str(value)])
    return args


def _effective_variant_signature(variant_args):
    """计算冻结主配置叠加变体后的有效签名，用于阻止重复对照。"""
    effective = {
        cli_name: 'none' if FINAL_CONFIG[key] is None else str(FINAL_CONFIG[key])
        for key, cli_name in FINAL_CONFIG_CLI_NAMES.items()
    }
    if len(variant_args) % 2:
        raise RuntimeError(f'消融参数必须为成对的 flag/value: {variant_args}')
    for index in range(0, len(variant_args), 2):
        effective[variant_args[index]] = str(variant_args[index + 1])

    mode = effective.get('--exploration-mode')
    if mode == 'noisy':
        effective['--model-type'] = 'set_dueling_noisy'
        effective['--eps-start'] = effective['--eps-end'] = '0.0'
    elif mode == 'none':
        effective['--model-type'] = 'set_dueling'
        effective['--eps-start'] = effective['--eps-end'] = '0.0'
    return tuple(sorted(effective.items()))


def _validate_family_variants(family):
    signatures = {}
    for name, variant_args in FAMILIES[family].items():
        signature = _effective_variant_signature(variant_args)
        if signature in signatures:
            raise RuntimeError(
                f'消融方案 {name} 与 {signatures[signature]} 的有效配置完全相同，请修复注册表后再运行'
            )
        signatures[signature] = name

    epsilon_args = dict(zip(
        FAMILIES.get('network_exploration', {}).get('epsilon_only', [])[::2],
        FAMILIES.get('network_exploration', {}).get('epsilon_only', [])[1::2],
    ))
    if family == 'network_exploration' and float(epsilon_args.get('--eps-start', 0)) <= 0:
        raise RuntimeError('epsilon_only 必须显式设置大于0的 eps-start')

DISPLAY_NAMES = {
    'full_hybrid': '完整混合模型',
    'no_pretrain': '去掉预训练',
    'no_expert_warmup': '去掉专家示范',
    'no_aux_bce': '去掉辅助监督',
    'pretrain_rl': '预训练 + RL',
    'supervised_only': '仅监督学习',
    'pure_rl': '仅 RL',
    'minimal_weighted_fbeta': '最小奖励（默认）',
    'no_step_cost': '去掉步成本',
    'final_weighted_fbeta': '最终奖励（无步成本）',
    'with_step_cost': '加入固定步成本',
    'no_label_reweight': '去掉类别权重',
    'epsilon_only': 'Epsilon 探索',
    'noisy_only': 'NoisyNet 探索',
    'no_exploration': '无探索',
    'plain_bce': '普通 BCE',
    'class_balanced_bce': '类别平衡 BCE',
    'focal_bce': 'Focal BCE',
    'no_rare_per': '标准 PER（无稀有加权）',
    'rare_per': '标准 PER + 稀有加权',
    'min_actions_0': '最少 0 个标签',
    'min_actions_1': '最少 1 个标签',
    'min_actions_2': '最少 2 个标签',
}

FAMILY_TITLES = {
    'supervision_components': '监督、示范与 RL 组成消融',
    'reward_simplification': '奖励函数组成消融',
    'network_exploration': '网络与探索策略消融',
    'tail_supervised_loss': '长尾监督损失消融',
    'tail_rare_per': '长尾 Replay 策略消融',
    'stop_policy': '停止策略消融',
}


def _extract_metric(result, metric):
    if metric == 'rare_macro_f1':
        return float(result['band_metrics']['rare_1_4']['macro_f1'])
    return float(result['auto'][metric])


def _metric_specs(family):
    if family in ('tail_supervised_loss', 'tail_rare_per'):
        return [
            ('supported_macro_f1', '总体 Supported Macro-F1', True),
            ('rare_macro_f1', '稀有标签 Macro-F1（support 1-4）', True),
        ]
    if family == 'stop_policy':
        return [
            ('sample_f1', '样本 F1', True),
            ('miss_selection_rate', '漏选率', False),
        ]
    return [
        ('supported_macro_f1', 'Supported Macro-F1', True),
        ('sample_f1', '样本 F1', True),
    ]


def _collect_results(family, batch_dir):
    records = json.loads((batch_dir / 'manifest.json').read_text(encoding='utf-8'))
    values = {name: {} for name in FAMILIES[family]}
    for record in records:
        if record.get('status') != 'done':
            continue
        metrics_path = Path(record['output_dir']) / 'validation_metrics.json'
        if not metrics_path.exists():
            # 支持把 Windows 结果目录复制到 Mac 后重新绘图。
            metrics_path = (
                Path(batch_dir) / record['experiment'] /
                f'seed_{record["seed"]}' / 'validation_metrics.json'
            )
        if not metrics_path.exists():
            continue
        result = json.loads(metrics_path.read_text(encoding='utf-8'))
        if record['experiment'] not in values:
            continue
        for metric, _, _ in _metric_specs(family):
            values[record['experiment']].setdefault(metric, []).append(_extract_metric(result, metric))
    return values


def plot_comparison(family, batch_dir, output_path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:
        raise RuntimeError('结果绘图需要 matplotlib') from exc

    # 优先直接绑定字体文件，避免 macOS/Windows 的字体名称缓存导致中文显示为方框。
    preferred_font_files = [
        Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
        Path('/Library/Fonts/Arial Unicode.ttf'),
        Path('C:/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/simhei.ttf'),
    ]
    font_path = next((path for path in preferred_font_files if path.exists()), None)
    preferred_fonts = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
        'PingFang SC', 'Hiragino Sans GB', 'Heiti TC', 'Arial Unicode MS',
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in preferred_fonts if font in available_fonts), 'DejaVu Sans')
    cjk_font = FontProperties(fname=str(font_path)) if font_path else FontProperties(family=selected_font)
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    values = _collect_results(family, batch_dir)
    experiments = [name for name in FAMILIES[family] if any(values[name].values())]
    if not experiments:
        raise RuntimeError('没有成功完成的实验，无法生成对比图；请查看各 seed 的 subprocess.stdout.log')

    labels = [DISPLAY_NAMES.get(name, name) for name in experiments]
    y_positions = np.arange(len(experiments))
    height = max(5.0, 0.62 * len(experiments) + 2.0)
    figure, axes = plt.subplots(1, 2, figsize=(14, height))

    conclusions = []
    for axis_index, (metric, title, higher_is_better) in enumerate(_metric_specs(family)):
        means = np.asarray([
            np.mean(values[name].get(metric, [np.nan])) for name in experiments
        ], dtype=float)
        stds = np.asarray([
            np.std(values[name].get(metric, [np.nan])) for name in experiments
        ], dtype=float)
        best_index = int(np.nanargmax(means) if higher_is_better else np.nanargmin(means))
        colors = ['#4C78A8'] * len(experiments)
        colors[best_index] = '#2A9D6F'
        axes[axis_index].barh(y_positions, means, xerr=stds, color=colors, alpha=0.9, capsize=3)
        axes[axis_index].set_yticks(y_positions)
        axes[axis_index].set_yticklabels(labels if axis_index == 0 else [], fontproperties=cjk_font)
        axes[axis_index].invert_yaxis()
        axes[axis_index].set_xlabel('均值（误差线为跨 seed 标准差）', fontproperties=cjk_font)
        direction = '越高越好' if higher_is_better else '越低越好'
        axes[axis_index].set_title(f'{title} · {direction}', fontproperties=cjk_font)
        axes[axis_index].grid(axis='x', alpha=0.2)
        offset = max(np.nanmax(np.abs(means)) * 0.015, 0.003)
        for index, value in enumerate(means):
            axes[axis_index].text(value + offset, index, f'{value:.3f}', va='center', fontsize=9)
        conclusions.append(f'{title} 最优：{labels[best_index]}（{means[best_index]:.4f}）')

    figure.suptitle(f'{FAMILY_TITLES[family]}（验证集选型）', fontsize=16, fontproperties=cjk_font)
    figure.text(
        0.5, 0.01,
        '绿色为该指标最优；若柱间差异小于误差线，不应判断存在稳定优势。',
        ha='center', fontproperties=cjk_font,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return conclusions


def run_clickable(family, source_dir):
    source_dir = Path(source_dir)
    _validate_family_variants(family)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_runs = len(FAMILIES[family]) * len(FORMAL_SEEDS)
    print('=' * 68)
    print(FAMILY_TITLES[family])
    print(f'冻结主配置版本：{FINAL_CONFIG_VERSION}；方案选择只使用验证集指标。')
    print(f'将自动运行 {len(FAMILIES[family])} 个方案 × {len(FORMAL_SEEDS)} 个随机种子，共 {total_runs} 次训练。')
    print('这是正式消融，可能需要数小时；运行过程中请不要关闭 VS Code 终端。')
    print('=' * 68, flush=True)
    batch_dir, records = run_family(
        family,
        FORMAL_SEEDS,
        extra_args=[
            *_frozen_final_config_args(),
            '--episodes', str(FORMAL_EPISODES),
            '--pretrain-epochs', str(FORMAL_PRETRAIN_EPOCHS),
            '--save-last-checkpoint', '0',
            '--device', 'auto',
            '--amp', '1',
            '--cuda-tf32', '1',
        ],
        run_id=run_id,
        stream_output=True,
    )
    failed = [record for record in records if record['status'] != 'done']
    if any(record['status'] == 'protocol_changed' for record in records):
        raise RuntimeError('检测到运行期间源码变化：本批次已作废，未生成对比图。请保持源码不变后重新点击运行。')
    if failed:
        print(f'警告：有 {len(failed)} 次训练失败，图中只统计成功结果。')
    batch_plot = batch_dir / 'comparison.png'
    conclusions = plot_comparison(family, batch_dir, batch_plot)
    local_plot = source_dir / 'latest_comparison.png'
    shutil.copyfile(batch_plot, local_plot)
    summary_path = source_dir / 'latest_result.txt'
    summary_path.write_text(
        '\n'.join([
            FAMILY_TITLES[family],
            f'结果目录：{batch_dir}',
            f'冻结主配置版本：{FINAL_CONFIG_VERSION}',
            '方案选择依据：验证集跨 seed 指标；测试集不参与方案选择。',
            *conclusions,
            '注意：误差线明显重叠时，不应仅凭均值判断组件重要性。',
        ]) + '\n',
        encoding='utf-8',
    )
    print('\n实验完成。')
    print(f'对比图：{local_plot}')
    print(f'文字结论：{summary_path}')
    return batch_dir


def run_tail_all(source_dir):
    run_clickable('tail_supervised_loss', Path(source_dir).parent / 'tail_supervised_loss')
    run_clickable('tail_rare_per', Path(source_dir).parent / 'tail_rare_per')
    print('长尾两阶段消融已全部完成，请分别查看两个子目录的 latest_comparison.png。')
