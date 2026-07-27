# -*- coding: utf-8 -*-
"""V1 消融公共注册表与子进程运行器。"""

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / 'main.py'
RESULTS = Path(__file__).resolve().parent / 'results'
DEFAULT_DATASET = ROOT.parents[2] / 'dataset' / 'lhz_data_cleaned.txt'

# 只监控会改变训练、推理或指标计算的实现。消融参数已在 batch_protocol.json
# 中冻结，因此绘图、README、预测工具和消融入口的修改不应作废正在运行的训练。
TRAINING_SOURCE_FILES = (
    'main.py',
    'final_config.py',
    'action.py',
    'constants.py',
    'data.py',
    'env.py',
    'evaluation.py',
    'memory.py',
    'metrics.py',
    'model.py',
    'state.py',
    'trainer.py',
    'utils.py',
)

FAMILIES = {
    'supervision_components': {
        'full_hybrid': ['--use-pretrain', '1', '--use-expert-warmup', '1', '--use-aux-bce', '1', '--aux-bce-weight', '0.05', '--run-rl', '1'],
        'no_pretrain': ['--use-pretrain', '0', '--use-expert-warmup', '1', '--use-aux-bce', '1', '--aux-bce-weight', '0.05', '--run-rl', '1'],
        'no_expert_warmup': ['--use-pretrain', '1', '--use-expert-warmup', '0', '--use-aux-bce', '1', '--aux-bce-weight', '0.05', '--run-rl', '1'],
        'no_aux_bce': ['--use-pretrain', '1', '--use-expert-warmup', '1', '--use-aux-bce', '0', '--aux-bce-weight', '0', '--run-rl', '1'],
        'pretrain_rl': ['--use-pretrain', '1', '--use-expert-warmup', '0', '--use-aux-bce', '0', '--aux-bce-weight', '0', '--run-rl', '1'],
        'supervised_only': ['--use-pretrain', '1', '--run-rl', '0', '--use-expert-warmup', '0', '--use-aux-bce', '0', '--aux-bce-weight', '0'],
        'pure_rl': ['--use-pretrain', '0', '--use-expert-warmup', '0', '--use-aux-bce', '0', '--aux-bce-weight', '0', '--run-rl', '1'],
    },
    'reward_simplification': {
        'final_weighted_fbeta': ['--step-cost', '0'],
        'with_step_cost': ['--step-cost', '0.02'],
        'no_label_reweight': ['--step-cost', '0', '--weight-min', '1', '--weight-max', '1'],
    },
    'network_exploration': {
        'epsilon_only': ['--model-type', 'set_dueling', '--exploration-mode', 'epsilon', '--eps-start', '0.7', '--eps-end', '0.02', '--eps-decay', '10000'],
        'noisy_only': ['--model-type', 'set_dueling_noisy', '--exploration-mode', 'noisy', '--eps-start', '0', '--eps-end', '0'],
        'no_exploration': ['--model-type', 'set_dueling', '--exploration-mode', 'none', '--eps-start', '0', '--eps-end', '0'],
    },
    'tail_supervised_loss': {
        'plain_bce': ['--supervised-loss', 'bce', '--rare-per-scale', '0'],
        'class_balanced_bce': ['--supervised-loss', 'class_balanced', '--rare-per-scale', '0'],
        'focal_bce': ['--supervised-loss', 'focal', '--rare-per-scale', '0'],
    },
    'tail_rare_per': {
        'no_rare_per': ['--rare-per-scale', '0'],
        'rare_per': ['--rare-per-scale', '0.5'],
    },
    'stop_policy': {
        'min_actions_0': ['--min-actions-before-stop', '0'],
        'min_actions_1': ['--min-actions-before-stop', '1'],
        'min_actions_2': ['--min-actions-before-stop', '2'],
    },
}


SUMMARY_METRICS = ('sample_f1', 'micro_f1', 'macro_f1', 'supported_macro_f1', 'exact_match',
                   'miss_selection_rate', 'zero_f1_count')


def _resolve_data_path(extra_args):
    """解析实际传给 main.py 的数据路径；点击运行时使用清洗数据默认值。"""
    args = list(extra_args or [])
    data_path = DEFAULT_DATASET
    for index, value in enumerate(args):
        if value == '--data-path' and index + 1 < len(args):
            data_path = Path(args[index + 1])
        elif value.startswith('--data-path='):
            data_path = Path(value.split('=', 1)[1])
    return Path(data_path).expanduser().resolve()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(data_path):
    """指纹化训练源码、消融注册表和数据，阻止批次内协议漂移。"""
    digest = hashlib.sha256()
    for filename in TRAINING_SOURCE_FILES:
        path = ROOT / filename
        if not path.exists():
            raise FileNotFoundError(f'训练源码缺失: {path}')
        digest.update(filename.encode('utf-8'))
        digest.update(path.read_bytes())
    registry_path = Path(__file__).resolve()
    digest.update(b'ablations/common.py')
    digest.update(registry_path.read_bytes())
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'消融数据集缺失: {data_path}')
    digest.update(str(data_path).encode('utf-8'))
    digest.update(data_path.read_bytes())
    return digest.hexdigest()


def _load_metrics(output_dir, split):
    path = output_dir / f'{split}_metrics.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8')).get('auto', {})


def _write_split_summaries(batch_dir, records, split):
    """从结构化 JSON 汇总，而非解析日志文本。"""
    detailed_rows = []
    for record in records:
        row = {'experiment': record['experiment'], 'seed': record['seed'], 'status': record['status']}
        if record['status'] == 'done':
            row.update(_load_metrics(Path(record['output_dir']), split))
        detailed_rows.append(row)
    fields = ['experiment', 'seed', 'status'] + list(SUMMARY_METRICS)
    with (batch_dir / f'per_seed_{split}_metrics.csv').open('w', encoding='utf-8-sig', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_rows = []
    names = sorted({row['experiment'] for row in detailed_rows})
    for name in names:
        rows = [row for row in detailed_rows if row['experiment'] == name and row['status'] == 'done']
        row = {'experiment': name, 'completed_seeds': len(rows)}
        for metric in SUMMARY_METRICS:
            values = [float(item[metric]) for item in rows if metric in item]
            row[f'{metric}_mean'] = sum(values) / len(values) if values else ''
            row[f'{metric}_std'] = (sum((value - row[f'{metric}_mean']) ** 2 for value in values) / len(values)) ** 0.5 if values else ''
        summary_rows.append(row)
    summary_fields = ['experiment', 'completed_seeds'] + [
        field for metric in SUMMARY_METRICS for field in (f'{metric}_mean', f'{metric}_std')
    ]
    with (batch_dir / f'cross_seed_{split}_summary.csv').open('w', encoding='utf-8-sig', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_summaries(batch_dir, records):
    _write_split_summaries(batch_dir, records, 'validation')
    _write_split_summaries(batch_dir, records, 'test')


def run_family(family, seeds, extra_args=None, run_id=None, dry_run=False, stream_output=False):
    if family not in FAMILIES:
        raise ValueError(f'未知消融类别: {family}')
    run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = RESULTS / family / run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    data_path = _resolve_data_path(extra_args)
    data_sha256 = _file_sha256(data_path)
    frozen_fingerprint = _source_fingerprint(data_path)
    protocol = {
        'family': family,
        'seeds': list(seeds),
        'extra_args': list(extra_args or []),
        'variants': FAMILIES[family],
        'fingerprinted_training_files': [*TRAINING_SOURCE_FILES, 'ablations/common.py'],
        'data_path': str(data_path),
        'data_sha256': data_sha256,
        'source_sha256': frozen_fingerprint,
    }
    (batch_dir / 'batch_protocol.json').write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    records = []
    protocol_changed = False
    for experiment, variant_args in FAMILIES[family].items():
        for seed in seeds:
            output_dir = batch_dir / experiment / f'seed_{seed}'
            # 公共预算参数在前，变体参数在后，防止调用者无意覆盖单因素消融的设置。
            # 显式传入已取指纹的绝对路径，确保协议记录、数据指纹和子进程实际读取的是同一个文件。
            command = (
                [sys.executable, str(MAIN), '--seed', str(seed), '--output-dir', str(output_dir)]
                + list(extra_args or [])
                + ['--data-path', str(data_path)]
                + variant_args
            )
            record = {
                'experiment': experiment,
                'seed': seed,
                'command': command,
                'output_dir': str(output_dir),
                'source_sha256': frozen_fingerprint,
            }
            if _source_fingerprint(data_path) != frozen_fingerprint:
                record['status'] = 'protocol_changed'
                records.append(record)
                protocol_changed = True
                print('错误：检测到消融运行期间源码发生变化，本批次已停止，不能用于正式比较。')
                break
            if dry_run:
                record['status'] = 'dry_run'
            else:
                (output_dir / 'subprocess.stdout.log').parent.mkdir(parents=True, exist_ok=True)
                log_path = output_dir / 'subprocess.stdout.log'
                if stream_output:
                    print(f'\n开始: {experiment}, seed={seed}', flush=True)
                    with log_path.open('w', encoding='utf-8') as log_file:
                        process = subprocess.Popen(
                            command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1,
                        )
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            log_file.write(line)
                        returncode = process.wait()
                else:
                    completed = subprocess.run(command, text=True, capture_output=True)
                    log_path.write_text(completed.stdout + '\n' + completed.stderr, encoding='utf-8')
                    returncode = completed.returncode
                record['status'] = 'done' if returncode == 0 else 'failed'
                record['returncode'] = returncode
                if _source_fingerprint(data_path) != frozen_fingerprint:
                    record['status'] = 'protocol_changed'
                    protocol_changed = True
                    print('错误：本次训练期间源码发生变化，本批次已停止，不能用于正式比较。')
            records.append(record)
            (batch_dir / 'manifest.json').write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            if protocol_changed:
                break
        if protocol_changed:
            break
    (batch_dir / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    if not dry_run:
        _write_summaries(batch_dir, records)
    return batch_dir, records
