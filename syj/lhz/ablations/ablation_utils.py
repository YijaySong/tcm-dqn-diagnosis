# -*- coding: utf-8 -*-
"""LHZ消融实验通用运行工具。

提供单个实验的安全运行封装：
1. 在独立结果目录中保存日志和实验产物。
2. 运行前备份`syj/lhz/mymodel/dqn_model.pth`和`syj/lhz/mymodel/split_data/`。
3. 运行后复制本次实验产物到结果目录，并恢复运行前文件状态，避免覆盖已有模型。
"""

import argparse
import json
import os
import runpy
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ABLATIONS_DIR = Path(__file__).resolve().parent
LHZ_DIR = ABLATIONS_DIR.parent
MYMODEL_DIR = LHZ_DIR / 'mymodel_reward_v2'
MAIN_SCRIPT = MYMODEL_DIR / 'main.py'
RESULTS_DIR = ABLATIONS_DIR / 'results'
MODEL_PATH = MYMODEL_DIR / 'dqn_model.pth'
SPLIT_DATA_DIR = MYMODEL_DIR / 'split_data'

EXPERIMENTS = [
    {
        'name': 'full',
        'label': 'Full',
        'script': 'run_full.py',
        'description': '完整模型：监督预训练 + 专家轨迹预填充 + 后续RL训练',
        'main_args': ['-use_pretrain', '1', '-replay_warmup', '1'],
    },
    {
        'name': 'no_pretrain',
        'label': 'w/o Pretrain',
        'script': 'run_no_pretrain.py',
        'description': '无预训练：关闭监督预训练，保留专家轨迹预填充和后续RL训练',
        'main_args': ['-use_pretrain', '0', '-replay_warmup', '1'],
    },
    {
        'name': 'pretrain_only',
        'label': 'Pretrain Only',
        'script': 'run_pretrain_only.py',
        'description': '仅预训练：只做监督预训练，不做专家轨迹预填充和后续RL训练',
        'main_args': ['-use_pretrain', '1', '-replay_warmup', '0', '-episode', '0'],
    },
    {
        'name': 'no_expert_warmup',
        'label': 'w/o Expert Warmup',
        'script': 'run_no_expert_warmup.py',
        'description': '无专家轨迹预填充：保留监督预训练和后续RL训练，关闭ReplayMemory专家预填充',
        'main_args': ['-use_pretrain', '1', '-replay_warmup', '0'],
    },
    {
        'name': 'rl_only',
        'label': 'RL Only',
        'script': 'run_rl_only.py',
        'description': '纯RL：无监督预训练、无专家轨迹预填充，仅从随机网络开始后续RL训练',
        'main_args': ['-use_pretrain', '0', '-replay_warmup', '0'],
    },
]
EXPERIMENT_BY_NAME = {item['name']: item for item in EXPERIMENTS}


def current_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def make_output_dir(experiment_name, run_id=None):
    run_id = run_id or current_timestamp()
    output_dir = RESULTS_DIR / run_id / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@contextmanager
def preserve_lhz_outputs():
    """Restore main.py side-effect outputs after an experiment finishes."""
    backup_dir = Path(tempfile.mkdtemp(prefix='lhz_ablation_backup_'))
    backup_model = backup_dir / 'dqn_model.pth'
    backup_split_data = backup_dir / 'split_data'
    had_model = MODEL_PATH.exists()
    had_split_data = SPLIT_DATA_DIR.exists()

    if had_model:
        shutil.copy2(MODEL_PATH, backup_model)
    if had_split_data:
        shutil.copytree(SPLIT_DATA_DIR, backup_split_data)

    try:
        yield
    finally:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        if had_model:
            shutil.copy2(backup_model, MODEL_PATH)

        if SPLIT_DATA_DIR.exists():
            shutil.rmtree(SPLIT_DATA_DIR)
        if had_split_data:
            shutil.copytree(backup_split_data, SPLIT_DATA_DIR)

        shutil.rmtree(backup_dir, ignore_errors=True)


def copy_experiment_outputs(output_dir):
    """Copy outputs produced by main.py into the experiment result directory."""
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, output_dir / 'dqn_model.pth')
    if SPLIT_DATA_DIR.exists():
        target = output_dir / 'split_data'
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(SPLIT_DATA_DIR, target)


def run_experiment(experiment_name, run_id=None, dry_run=False, extra_main_args=None):
    """Run one ablation experiment safely and return its output directory."""
    if experiment_name not in EXPERIMENT_BY_NAME:
        known = ', '.join(sorted(EXPERIMENT_BY_NAME))
        raise ValueError(f'未知实验: {experiment_name}. 可选实验: {known}')

    experiment = EXPERIMENT_BY_NAME[experiment_name]
    output_dir = make_output_dir(experiment_name, run_id=run_id)
    main_args = list(experiment['main_args'])
    if extra_main_args:
        main_args.extend(extra_main_args)

    metadata = {
        'name': experiment['name'],
        'label': experiment['label'],
        'description': experiment['description'],
        'main_script': str(MAIN_SCRIPT),
        'main_args': main_args,
        'output_dir': str(output_dir),
    }
    (output_dir / 'experiment_config.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    command_text = ' '.join([sys.executable, str(MAIN_SCRIPT)] + main_args)
    if dry_run:
        print(f'[DRY-RUN] {experiment["label"]}: {command_text}')
        print(f'[DRY-RUN] 输出目录: {output_dir}')
        return output_dir

    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    inserted_path = False
    lhz_path = str(MYMODEL_DIR)

    print(f'========== 开始实验: {experiment["label"]} ==========')
    print(f'说明: {experiment["description"]}')
    print(f'输出目录: {output_dir}')
    print(f'主程序参数: {" ".join(main_args)}')

    with preserve_lhz_outputs():
        try:
            if lhz_path not in sys.path:
                sys.path.insert(0, lhz_path)
                inserted_path = True
            os.chdir(output_dir)
            sys.argv = [str(MAIN_SCRIPT)] + main_args
            runpy.run_path(str(MAIN_SCRIPT), run_name='__main__')
            copy_experiment_outputs(output_dir)
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
            if inserted_path:
                try:
                    sys.path.remove(lhz_path)
                except ValueError:
                    pass

    print(f'========== 实验完成: {experiment["label"]} ==========')
    print(f'结果已保存: {output_dir}')
    return output_dir


def wrapper_cli(experiment_name):
    parser = argparse.ArgumentParser(description=f'运行LHZ消融实验: {experiment_name}')
    parser.add_argument('--run-id', default=None, help='结果目录批次名；默认使用当前时间戳')
    parser.add_argument('--dry-run', action='store_true', help='只显示将要执行的命令，不启动训练')
    parser.add_argument(
        'main_args', nargs=argparse.REMAINDER,
        help='追加传给syj/lhz/mymodel_reward_v2/main.py的参数；如需使用，请放在 -- 后面，例如: -- -episode 5'
    )
    args = parser.parse_args()
    extra_main_args = args.main_args
    if extra_main_args and extra_main_args[0] == '--':
        extra_main_args = extra_main_args[1:]
    run_experiment(
        experiment_name,
        run_id=args.run_id,
        dry_run=args.dry_run,
        extra_main_args=extra_main_args,
    )
