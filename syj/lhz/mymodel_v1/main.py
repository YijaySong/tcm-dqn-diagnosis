# -*- coding: utf-8 -*-
"""mymodel_v1 训练入口：症状组隔离、验证选模、简化奖励、结构化产物。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from constants import resolve_device
from data import (build_split_manifest, compute_Se_supports, compute_Se_weights,
                  export_split_data, fit_training_schema, group_split_data,
                  load_raw_tcm_data, strip_sample_id, transform_split)
from env import Environment
from final_config import DECISION_STATUS, FINAL_CONFIG, FINAL_CONFIG_VERSION
from memory import ReplayMemory
from model import build_q_network
from trainer import DQNTrainer
from utils import (capture_rng_state, configure_accelerator, create_logger,
                   default_data_path, restore_rng_state, set_seed)


def optional_float(value):
    if value is None or str(value).strip().lower() in ('none', 'null'):
        return None
    return float(value)


def build_parser():
    final = FINAL_CONFIG
    parser = argparse.ArgumentParser(description='V1 标签驱动序贯多标签 DQN')
    parser.add_argument('--data-path', default=default_data_path())
    parser.add_argument('--output-dir', default=None, help='全部产物写入该目录；默认 runs/<timestamp>')
    parser.add_argument('--resume', default=None, help='从同一数据协议产生的 V1 last.pt 恢复 RL 训练')
    parser.add_argument('--device', default='auto',
                        help='auto/cpu/cuda/cuda:0；Windows NVIDIA 默认自动选择 CUDA')
    parser.add_argument('--amp', type=int, default=1, help='CUDA 上是否启用 FP16 混合精度')
    parser.add_argument('--cuda-tf32', type=int, default=1, help='支持的 NVIDIA GPU 上是否启用 TF32')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--seeds', default=None, help='逗号分隔多个训练随机种子；划分仍由 split-seed 固定')
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--max-se-num', type=int, default=0)
    parser.add_argument('--unknown-label-policy', choices=['drop_sample', 'error'], default='drop_sample')

    parser.add_argument('--episodes', '-episode', type=int, default=100)
    parser.add_argument('--batch-size', '-batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=final['gamma'])
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--nn-units', type=int, default=final['nn_units'])
    parser.add_argument('--nn-units2', type=int, default=final['nn_units2'])
    parser.add_argument('--dropout', type=float, default=final['dropout'])
    parser.add_argument('--model-type', choices=['set_dueling', 'set_dueling_noisy'], default=final['model_type'])
    parser.add_argument('--exploration-mode', choices=['epsilon', 'noisy', 'none'], default=final['exploration_mode'])
    parser.add_argument('--eps-start', type=float, default=final['eps_start'])
    parser.add_argument('--eps-end', type=float, default=final['eps_end'])
    parser.add_argument('--eps-decay', type=int, default=final['eps_decay'])
    parser.add_argument('--min-actions-before-stop', type=int, default=final['min_actions_before_stop'])
    parser.add_argument('--stop-margin-threshold', type=optional_float, default=final['stop_margin_threshold'])

    parser.add_argument('--reward-beta', type=float, default=final['reward_beta'])
    parser.add_argument('--step-cost', type=float, default=final['step_cost'])
    parser.add_argument('--stop-utility-scale', type=float, default=final['stop_utility_scale'])
    parser.add_argument('--weight-min', type=float, default=final['weight_min'])
    parser.add_argument('--weight-max', type=float, default=final['weight_max'])
    parser.add_argument('--weight-power', type=float, default=final['weight_power'])

    parser.add_argument('--use-pretrain', type=int, default=final['use_pretrain'])
    parser.add_argument('--pretrain-epochs', type=int, default=final['pretrain_epochs'])
    parser.add_argument('--pretrain-lr', type=float, default=final['pretrain_lr'])
    parser.add_argument('--pretrain-num-orders', type=int, default=final['pretrain_num_orders'])
    parser.add_argument('--supervised-loss', choices=['bce', 'class_balanced', 'focal'], default=final['supervised_loss'])
    parser.add_argument('--focal-gamma', type=float, default=final['focal_gamma'])
    parser.add_argument('--use-expert-warmup', type=int, default=final['use_expert_warmup'])
    parser.add_argument('--replay-capacity', type=int, default=0, help='0时自动容纳所有专家轨迹')
    parser.add_argument('--use-per', type=int, default=final['use_per'])
    parser.add_argument('--per-alpha', type=float, default=0.6)
    parser.add_argument('--per-beta-start', type=float, default=0.4)
    parser.add_argument('--per-beta-frames', type=int, default=5000)
    parser.add_argument('--rare-per-scale', type=float, default=final['rare_per_scale'])
    parser.add_argument('--use-aux-bce', type=int, default=final['use_aux_bce'])
    parser.add_argument('--aux-bce-weight', type=float, default=final['aux_bce_weight'])
    parser.add_argument('--aux-pos-weight-max', type=float, default=8.0)
    parser.add_argument('--pretrain-pos-weight-max', type=float, default=18.0)
    parser.add_argument('--run-rl', type=int, default=final['run_rl'])
    parser.add_argument('--save-last-checkpoint', type=int, default=1,
                        help='是否保存含 replay 的 last.pt；一键消融默认关闭以节省磁盘和写盘时间')

    parser.add_argument('--selection-metric', choices=['supported_macro_f1', 'sample_f1', 'macro_f1'], default='supported_macro_f1')
    parser.add_argument('--patience', type=int, default=0,
                        help='验证早停耐心值；默认0表示关闭早停并完整运行 episodes 轮')
    parser.add_argument('--min-delta', type=float, default=0.001)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def _resolved_exploration(args):
    if args.exploration_mode == 'noisy':
        args.model_type, args.eps_start, args.eps_end = 'set_dueling_noisy', 0.0, 0.0
    elif args.exploration_mode == 'none':
        args.model_type, args.eps_start, args.eps_end = 'set_dueling', 0.0, 0.0
    elif args.model_type == 'set_dueling_noisy':
        raise ValueError('epsilon 模式不能同时使用 NoisyNet；请使用 --exploration-mode noisy')


def _align_resume_arguments(args):
    """恢复时锁定原 run 的实验协议，避免误用不同网络或划分继续训练。"""
    if not args.resume:
        return
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    config = checkpoint.get('config')
    if checkpoint.get('checkpoint_version') != 3 or checkpoint.get('kind') != 'last' or not isinstance(config, dict):
        raise ValueError('--resume 必须指向本版本生成的 last.pt')
    operational = {
        'output_dir', 'resume', 'episodes', 'patience', 'min_delta', 'dry_run',
        'seed', 'seeds', 'device', 'amp', 'cuda_tf32',
    }
    for key, value in config.items():
        if key not in operational and hasattr(args, key):
            setattr(args, key, value)


def _prepare_data(args, logger, split_dir):
    raw = load_raw_tcm_data(args.data_path, logger)
    raw_splits = group_split_data(raw, args.split_seed, args.train_ratio, args.val_ratio, args.test_ratio, logger)
    schema = fit_training_schema(raw_splits['train'], args.max_se_num, logger)
    transformed, transform_stats = {}, {}
    for name, records in raw_splits.items():
        transformed[name], transform_stats[name] = transform_split(records, schema, name, args.unknown_label_policy)
    manifest = build_split_manifest(
        args.data_path, args.split_seed,
        (args.train_ratio, args.val_ratio, args.test_ratio), raw,
        raw_splits, schema, transform_stats,
    )
    export_split_data(transformed, split_dir, manifest, logger)
    return transformed, schema, manifest


def _save_checkpoint(path, trainer, args, schema, manifest, epoch, validation_metrics, kind):
    payload = {
        'checkpoint_version': 3, 'kind': kind, 'epoch': epoch,
        'policy_state_dict': trainer.policy_net.state_dict(),
        'target_state_dict': trainer.target_net.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'model_type': args.model_type, 'nn_units': args.nn_units, 'nn_units2': args.nn_units2,
        'dropout': args.dropout, 'symptoms': schema['symptoms'], 'Se': schema['labels'],
        'training_support': schema['train_label_support'], 'config': vars(args),
        'split_manifest': manifest, 'validation_metrics': validation_metrics,
        'rng_state': capture_rng_state(),
    }
    if kind == 'last':
        payload['training_state'] = trainer.training_state()
    torch.save(payload, path)


def _resume_from_checkpoint(path, trainer, args, schema, manifest, logger):
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    if checkpoint.get('checkpoint_version') != 3 or checkpoint.get('kind') != 'last':
        raise ValueError('--resume 必须指向本版本生成的 last.pt')
    old_manifest = checkpoint.get('split_manifest', {})
    if old_manifest.get('data_sha256') != manifest.get('data_sha256'):
        raise ValueError('恢复 checkpoint 的数据文件与当前数据不一致')
    if old_manifest.get('split_seed') != manifest.get('split_seed'):
        raise ValueError('恢复 checkpoint 的 split-seed 与当前设置不一致')
    if checkpoint.get('Se') != schema['labels'] or checkpoint.get('symptoms') != schema['symptoms']:
        raise ValueError('恢复 checkpoint 的训练词表与当前词表不一致')
    trainer.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    trainer.target_net.load_state_dict(checkpoint['target_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.restore_training_state(checkpoint.get('training_state'))
    restore_rng_state(checkpoint.get('rng_state'))
    logger.info(f'已恢复 checkpoint: {path}, epoch={checkpoint.get("epoch")}, replay={len(trainer.memory)}')


def run_one(args, seed, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir / 'train.log')
    training_device = resolve_device(args.device)
    accelerator = configure_accelerator(training_device, bool(args.cuda_tf32))
    set_seed(seed)
    amp_enabled = bool(args.amp) and training_device.type == 'cuda'
    logger.info(
        f'V1 device={training_device}, gpu={accelerator["gpu_name"]}, '
        f'CUDA={accelerator["cuda_version"]}, AMP={amp_enabled}, TF32={accelerator["tf32"]}, '
        f'train_seed={seed}, split_seed={args.split_seed}'
    )
    logger.info(f'最终配置版本={FINAL_CONFIG_VERSION}, 消融决策状态={DECISION_STATUS}')
    split_data, schema, manifest = _prepare_data(args, logger, output_dir / 'splits')
    train_data = strip_sample_id(split_data['train'])
    validation_data = strip_sample_id(split_data['validation'])
    test_data = strip_sample_id(split_data['test'])
    if not train_data or not validation_data or not test_data:
        raise ValueError('症状组划分后存在空集合；请调整比例或数据')
    env = Environment(schema['symptoms'], schema['labels'])
    env.configure_reward(args.reward_beta, args.step_cost, args.stop_utility_scale, args.gamma)
    env.set_Se_weights(compute_Se_weights(train_data, env, logger, args.weight_min, args.weight_max, args.weight_power))
    env.set_Se_supports(compute_Se_supports(train_data, env, logger))
    state_length, action_count = len(env.state_space), len(env.action_space)
    policy = build_q_network(args.model_type, state_length, action_count, args.nn_units, args.nn_units2, args.dropout).to(training_device)
    target = build_q_network(args.model_type, state_length, action_count, args.nn_units, args.nn_units2, args.dropout).to(training_device)
    target.load_state_dict(policy.state_dict())
    estimated_expert = sum((len(labels) + 1) * max(1, args.pretrain_num_orders) for _, labels in train_data)
    capacity = args.replay_capacity or max(estimated_expert, args.batch_size * 4)
    if args.use_expert_warmup and capacity < estimated_expert:
        raise ValueError(f'replay-capacity={capacity} 不能容纳专家轨迹预估={estimated_expert}')
    memory = ReplayMemory(capacity, prioritized=bool(args.use_per), alpha=args.per_alpha)
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    trainer = DQNTrainer(
        env, train_data, validation_data, policy, target, memory, optimizer, training_device, logger,
        args.batch_size, args.gamma, args.tau, args.eps_start, args.eps_end, args.eps_decay,
        args.n_step, args.min_actions_before_stop, bool(args.use_aux_bce), args.aux_bce_weight,
        args.aux_pos_weight_max, args.pretrain_pos_weight_max, args.supervised_loss, args.focal_gamma,
        bool(args.use_per), args.per_beta_start, args.per_beta_frames, args.rare_per_scale,
        args.stop_margin_threshold, amp_enabled,
    )
    if args.resume:
        _resume_from_checkpoint(args.resume, trainer, args, schema, manifest, logger)
    resolved = vars(args).copy()
    resolved.update({
        'final_config_version': FINAL_CONFIG_VERSION,
        'final_config_snapshot': FINAL_CONFIG,
        'decision_status': DECISION_STATUS,
        'train_seed': seed,
        'estimated_expert_transitions': estimated_expert,
        'replay_capacity': capacity,
        'resolved_device': str(training_device),
        'accelerator': accelerator,
        'amp_enabled': amp_enabled,
    })
    (output_dir / 'resolved_config.json').write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.dry_run:
        logger.info(f'DRY RUN: train={len(train_data)}, val={len(validation_data)}, test={len(test_data)}, replay={capacity}')
        return {'status': 'dry_run', 'manifest': manifest}
    if args.use_pretrain and not args.resume:
        trainer.pretrain_supervised(args.pretrain_epochs, args.pretrain_lr, args.pretrain_num_orders)
    if args.use_expert_warmup and not args.resume:
        trainer.warmup_replay_with_expert(args.pretrain_num_orders)

    def checkpoint_callback(kind, epoch, validation):
        if kind == 'last' and not args.save_last_checkpoint:
            return
        _save_checkpoint(output_dir / f'{kind}.pt', trainer, args, schema, manifest, epoch, validation, kind)

    if args.run_rl and args.episodes > 0:
        fit_result = trainer.fit(args.episodes, args.patience, args.min_delta, args.selection_metric, checkpoint_callback)
    else:
        validation = trainer.evaluate(validation_data, '验证集')['auto']
        checkpoint_callback('best', -1, validation)
        checkpoint_callback('last', -1, validation)
        fit_result = {'best_epoch': -1, 'best_value': validation.get(args.selection_metric, validation['sample_f1']), 'history': []}
    validation_result = trainer.evaluate(validation_data, '最佳模型验证集')
    test_result = trainer.evaluate(test_data, '最终测试集')
    (output_dir / 'validation_metrics.json').write_text(json.dumps(validation_result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'test_metrics.json').write_text(json.dumps(test_result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'history.json').write_text(json.dumps(fit_result['history'], ensure_ascii=False, indent=2), encoding='utf-8')
    import csv
    for filename, result in (('validation_label_metrics.csv', validation_result), ('test_label_metrics.csv', test_result)):
        rows = result['label_metrics']
        if rows:
            with (output_dir / filename).open('w', encoding='utf-8-sig', newline='') as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return {'status': 'done', 'fit': fit_result, 'validation': validation_result, 'test': test_result, 'manifest': manifest}


def main():
    args = build_parser().parse_args()
    _align_resume_arguments(args)
    _resolved_exploration(args)
    root = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    seeds = [args.seed] if not args.seeds else [int(value.strip()) for value in args.seeds.split(',') if value.strip()]
    root.mkdir(parents=True, exist_ok=True)
    all_results = []
    multi_seed = len(seeds) > 1
    for seed in seeds:
        seed_output_dir = root / f'seed_{seed}' if multi_seed else root
        all_results.append({'seed': seed, **run_one(args, seed, seed_output_dir)})
    (root / 'all_seed_results.json').write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
    if len(all_results) > 1 and not args.dry_run:
        values = [item['test']['auto'] for item in all_results if item['status'] == 'done']
        summary = {key: {'mean': sum(row[key] for row in values) / len(values),
                         'std': float(np.std([row[key] for row in values]))}
                   for key in ('sample_f1', 'micro_f1', 'macro_f1', 'supported_macro_f1', 'exact_match')}
        (root / 'cross_seed_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'V1 结果目录: {root}')


if __name__ == '__main__':
    main()
