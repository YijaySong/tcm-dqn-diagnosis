# -*- coding: utf-8 -*-
"""主程序入口文件。

解析命令行参数，加载lhz_data.txt数据集，初始化环境、DQN网络、经验池和训练器，
然后依次完成训练、测试集评估、模型保存和示例预测。
"""

import os
from argparse import ArgumentParser
from datetime import datetime

import torch.optim as optim

from constants import device
from data import compute_Se_supports, compute_Se_weights, export_split_data, get_tcm_data, stratified_split_data, strip_sample_id
from env import Environment
from memory import ReplayMemory
from model import build_q_network
from trainer import DQNTrainer, save_checkpoint
from utils import create_logger, default_data_path, set_seed


def build_parser():
    parser = ArgumentParser(description="DQN Intelligent Syndrome Differentiation")
    parser.add_argument("-episode", type=int, dest="num_episodes", default=50)
    parser.add_argument("-batch", type=int, dest="batch_size", default=64)
    parser.add_argument("-mem_capacity", type=int, dest="mem_capacity", default=10000)
    parser.add_argument("-gamma", type=float, dest="gamma", default=0.95)
    parser.add_argument("-eps_start", type=float, dest="eps_start", default=0.7)
    parser.add_argument("-eps_end", type=float, dest="eps_end", default=0.02)
    parser.add_argument("-eps_decay", type=int, dest="eps_decay", default=10000)
    parser.add_argument("-tau", type=float, dest="tau", default=0.01)
    parser.add_argument("-lr", type=float, dest="lr", default=0.0003)
    parser.add_argument("-weight_decay", type=float, dest="weight_decay", default=1e-4)
    parser.add_argument("-nn_units", type=int, dest="nn_units", default=256)
    parser.add_argument("-nn_units2", type=int, dest="nn_units2", default=128)
    parser.add_argument("-dropout", type=float, dest="dropout", default=0.15)
    parser.add_argument(
        "-model_type", choices=['dqn_legacy', 'dqn', 'dueling', 'dueling_residual', 'set_dueling', 'set_dueling_noisy'],
        dest="model_type", default='set_dueling_noisy'
    )
    parser.add_argument("-aux_supervised_weight", type=float, dest="aux_supervised_weight", default=0.03)
    parser.add_argument("-aux_supervised_start", type=float, dest="aux_supervised_start", default=0.15)
    parser.add_argument("-aux_pos_weight_max", type=float, dest="aux_pos_weight_max", default=8.0)
    parser.add_argument("-pretrain_pos_weight_max", type=float, dest="pretrain_pos_weight_max", default=18.0)
    parser.add_argument("-pretrain_anchor_weight", type=float, dest="pretrain_anchor_weight", default=0.0)
    parser.add_argument("-pretrain_all_permutations", type=int, dest="pretrain_all_permutations", default=2)
    parser.add_argument("-seed", type=int, dest="seed", default=9)
    parser.add_argument("-max_se_num", type=int, dest="max_Se_num", default=0)
    parser.add_argument("-use_pretrain", type=int, dest="use_pretrain", default=1)
    parser.add_argument("-pretrain_epochs", type=int, dest="pretrain_epochs", default=5)  # 降低到5，让预训练"记不住"稀有标签，给好奇心机制发挥空间
    parser.add_argument("-pretrain_lr", type=float, dest="pretrain_lr", default=0.001)
    parser.add_argument("-replay_warmup", type=int, dest="replay_warmup", default=1)
    parser.add_argument("-test_ratio", type=float, dest="test_ratio", default=0.2)
    parser.add_argument(
        "-reward_mode",
        choices=['legacy', 'potential_f1', 'macro_balanced', 'tail_cost_curiosity'],
        dest="reward_mode", default='tail_cost_curiosity'
    )
    parser.add_argument("-step_penalty", type=float, dest="step_penalty", default=None)
    parser.add_argument("-false_positive_penalty", type=float, dest="false_positive_penalty", default=None)
    parser.add_argument("-false_positive_penalty_start", type=float, dest="false_positive_penalty_start", default=None)
    parser.add_argument("-over_select_penalty", type=float, dest="over_select_penalty", default=None)
    parser.add_argument("-potential_baseline", type=float, dest="potential_baseline", default=None)
    parser.add_argument("-cardinality_penalty", type=float, dest="cardinality_penalty", default=None)
    parser.add_argument("-terminal_f1_scale", type=float, dest="terminal_f1_scale", default=None)
    parser.add_argument("-exact_match_bonus", type=float, dest="exact_match_bonus", default=None)
    parser.add_argument("-stop_fn_penalty", type=float, dest="stop_fn_penalty", default=None)
    parser.add_argument("-stop_fp_penalty", type=float, dest="stop_fp_penalty", default=None)
    parser.add_argument("-terminal_cardinality_penalty", type=float, dest="terminal_cardinality_penalty", default=None)
    parser.add_argument("-tp_weight_scale", type=float, dest="tp_weight_scale", default=None)
    parser.add_argument("-fp_weight_scale", type=float, dest="fp_weight_scale", default=None)
    parser.add_argument("-rare_tp_bonus", type=float, dest="rare_tp_bonus", default=None)
    parser.add_argument("-rare_tp_bonus_start", type=float, dest="rare_tp_bonus_start", default=None)
    parser.add_argument("-rare_tp_bonus_end", type=float, dest="rare_tp_bonus_end", default=None)
    parser.add_argument("-rare_fn_penalty", type=float, dest="rare_fn_penalty", default=None)
    parser.add_argument("-balanced_beta", type=float, dest="balanced_beta", default=None)
    parser.add_argument("-terminal_sample_f1_weight", type=float, dest="terminal_sample_f1_weight", default=None)
    parser.add_argument("-terminal_balanced_f1_weight", type=float, dest="terminal_balanced_f1_weight", default=None)
    parser.add_argument("-uncertainty_tp_bonus", type=float, dest="uncertainty_tp_bonus", default=None)
    parser.add_argument("-pretrain_miss_tp_bonus", type=float, dest="pretrain_miss_tp_bonus", default=None)
    parser.add_argument("-uncertainty_potential_scale", type=float, dest="uncertainty_potential_scale", default=None)
    parser.add_argument("-terminal_tail_recall_bonus", type=float, dest="terminal_tail_recall_bonus", default=None)
    parser.add_argument("-pretrain_miss_fn_penalty", type=float, dest="pretrain_miss_fn_penalty", default=None)
    parser.add_argument("-exploration_tp_bonus", type=float, dest="exploration_tp_bonus", default=None)  # 探索奖励
    parser.add_argument("-f1_log_gain", type=float, dest="f1_log_gain", default=None)
    parser.add_argument("-over_select_penalty_power", type=float, dest="over_select_penalty_power", default=None)
    parser.add_argument("-terminal_under_select_penalty", type=float, dest="terminal_under_select_penalty", default=None)
    parser.add_argument("-support_confidence_min", type=float, dest="support_confidence_min", default=None)
    parser.add_argument("-support_confidence_k", type=float, dest="support_confidence_k", default=None)
    parser.add_argument("-Se_weight_method", choices=['power', 'log'], dest="Se_weight_method", default='log')
    parser.add_argument("-Se_weight_log_scale", type=float, dest="Se_weight_log_scale", default=0.65)
    parser.add_argument("-target_update_strategy", choices=['hard', 'soft'], dest="target_update_strategy", default='soft')
    parser.add_argument("-target_update_interval", type=int, dest="target_update_interval", default=1)
    parser.add_argument("-use_per", type=int, dest="use_per", default=1)
    parser.add_argument("-per_alpha", type=float, dest="per_alpha", default=0.6)
    parser.add_argument("-per_beta_start", type=float, dest="per_beta_start", default=0.4)
    parser.add_argument("-per_beta_frames", type=int, dest="per_beta_frames", default=5000)
    parser.add_argument("-priority_clip", type=float, dest="priority_clip", default=10.0)
    parser.add_argument("-optimize_interval", type=int, dest="optimize_interval", default=5)
    parser.add_argument("-n_step", type=int, dest="n_step", default=3)
    parser.add_argument("-use_curriculum", type=int, dest="use_curriculum", default=1)
    parser.add_argument("-curriculum_stage1", type=float, dest="curriculum_stage1", default=0.3)
    parser.add_argument("-curriculum_stage2", type=float, dest="curriculum_stage2", default=0.6)
    parser.add_argument("-min_actions_before_stop", type=int, dest="min_actions_before_stop", default=2)
    parser.add_argument("-stop_margin_threshold", type=float, dest="stop_margin_threshold", default=None)
    parser.add_argument("-rare_priority_scale", type=float, dest="rare_priority_scale", default=1.1)
    parser.add_argument("-pretrain_miss_priority_scale", type=float, dest="pretrain_miss_priority_scale", default=0.7)
    parser.add_argument("-refresh_hints_interval", type=int, dest="refresh_hints_interval", default=10)  # 每N次迭代更新一次pretrain_hints，让好奇心机制适应网络变化
    parser.add_argument("-Se_weight_min", type=float, dest="Se_weight_min", default=0.6)
    parser.add_argument("-Se_weight_max", type=float, dest="Se_weight_max", default=6.0)
    parser.add_argument("-Se_weight_power", type=float, dest="Se_weight_power", default=0.85)
    parser.add_argument("-patience", type=int, dest="patience", default=15, help="保留兼容参数；当前版本不再用测试集早停")
    parser.add_argument("-min_delta", type=float, dest="min_delta", default=0.001, help="保留兼容参数；当前版本不再用测试集早停")
    return parser


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)

    lr_name = str(args.lr).replace('0.', '')
    now = datetime.now()
    time_sign = f"{now.year}年{now.month}月{now.day}日{now.hour}时{now.minute}分"

    logger = create_logger(f"dqn_LR{lr_name}_{args.nn_units}_{args.nn_units2}units_{time_sign}.log")
    logger.info(f"可用设备:{device}")
    logger.info(f"随机种子:{args.seed}")

    data_path = default_data_path()
    tcm_data, symptoms, Se, max_Se_len = get_tcm_data(data_path, max_Se_num=args.max_Se_num, logger=logger)
    env = Environment(symptoms, Se)
    reward_kwargs = {'mode': args.reward_mode, 'shaping_gamma': args.gamma}
    for key in (
        'step_penalty', 'false_positive_penalty', 'false_positive_penalty_start',
        'over_select_penalty', 'potential_baseline', 'cardinality_penalty',
        'terminal_f1_scale', 'exact_match_bonus', 'stop_fn_penalty', 'stop_fp_penalty',
        'terminal_cardinality_penalty', 'tp_weight_scale', 'fp_weight_scale',
        'rare_tp_bonus', 'rare_tp_bonus_start', 'rare_tp_bonus_end', 'rare_fn_penalty',
        'balanced_beta', 'terminal_sample_f1_weight', 'terminal_balanced_f1_weight',
        'uncertainty_tp_bonus', 'pretrain_miss_tp_bonus', 'uncertainty_potential_scale',
        'terminal_tail_recall_bonus', 'pretrain_miss_fn_penalty', 'exploration_tp_bonus',
        'f1_log_gain', 'over_select_penalty_power', 'terminal_under_select_penalty',
        'support_confidence_min', 'support_confidence_k',
    ):
        value = getattr(args, key)
        if value is not None:
            reward_kwargs[key] = value
    env.configure_reward(**reward_kwargs)

    training_tcm_data_raw, test_tcm_data_raw = stratified_split_data(
        tcm_data, args.seed, test_ratio=args.test_ratio, logger=logger
    )
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_data')
    export_split_data(
        training_tcm_data_raw, test_tcm_data_raw, export_dir,
        args.seed, args.test_ratio, logger=logger
    )
    training_tcm_data = strip_sample_id(training_tcm_data_raw)
    test_tcm_data = strip_sample_id(test_tcm_data_raw)
    env.set_Se_weights(compute_Se_weights(
        training_tcm_data, env, logger=logger,
        min_weight=args.Se_weight_min, max_weight=args.Se_weight_max,
        power=args.Se_weight_power, method=args.Se_weight_method,
        log_scale=args.Se_weight_log_scale,
    ))
    env.set_Se_supports(compute_Se_supports(training_tcm_data, env, logger=logger))

    logger.info(f"训练数据数量:{len(training_tcm_data)}, 测试数据数量:{len(test_tcm_data)}, 测试集比例:{args.test_ratio:.2f}")
    logger.info(f"最大真实证候要素数:{max_Se_len}")

    state_vector_len = len(env.state_space)
    n_actions = len(env.action_space)
    logger.info(f"状态向量长度:{state_vector_len}, 动作数:{n_actions}")

    policy_net = build_q_network(args.model_type, state_vector_len, n_actions, args.nn_units, args.nn_units2, args.dropout).to(device)
    target_net = build_q_network(args.model_type, state_vector_len, n_actions, args.nn_units, args.nn_units2, args.dropout).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ReplayMemory(args.mem_capacity, prioritized=bool(args.use_per), alpha=args.per_alpha)
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    logger.info(
        f"**设置**: batch={args.batch_size}, mem={args.mem_capacity}, gamma={args.gamma}, "
        f"eps={args.eps_start}->{args.eps_end}, eps_decay={args.eps_decay}, tau={args.tau}, "
        f"target_update={args.target_update_strategy}/{args.target_update_interval}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, hidden={args.nn_units}/{args.nn_units2}, "
        f"dropout={args.dropout}, model_type={args.model_type}, pretrain={args.use_pretrain}, "
        f"warmup={args.replay_warmup}, aux_supervised_weight={args.aux_supervised_start}->{args.aux_supervised_weight}, "
        f"pretrain_all_permutations={args.pretrain_all_permutations}, test_ratio={args.test_ratio}, "
        f"Se_weight={args.Se_weight_method}:{args.Se_weight_min}-{args.Se_weight_max}/"
        f"p{args.Se_weight_power}/log{args.Se_weight_log_scale}, "
        f"reward={args.reward_mode}, use_per={args.use_per}, per_alpha={args.per_alpha}, "
        f"per_beta={args.per_beta_start}->1.0/{args.per_beta_frames}, n_step={args.n_step}, "
        f"curriculum={args.use_curriculum}, stop_margin_threshold={args.stop_margin_threshold}, "
        f"pretrain_miss_priority_scale={args.pretrain_miss_priority_scale}"
    )
    logger.info(f"奖励配置: {env.reward_config}")

    trainer = DQNTrainer(
        env, training_tcm_data, test_tcm_data, policy_net, target_net, memory,
        optimizer, device, logger, args.batch_size, args.gamma, args.tau,
        args.weight_decay, args.eps_start, args.eps_end, args.eps_decay, n_actions,
        args.aux_supervised_weight, args.pretrain_all_permutations,
        args.target_update_strategy, args.target_update_interval,
        args.per_beta_start, args.per_beta_frames, args.priority_clip,
        args.optimize_interval, args.n_step, args.aux_supervised_start,
        args.aux_pos_weight_max, args.pretrain_pos_weight_max, args.pretrain_anchor_weight,
        bool(args.use_curriculum), args.curriculum_stage1, args.curriculum_stage2,
        args.min_actions_before_stop, args.stop_margin_threshold, args.rare_priority_scale,
        args.pretrain_miss_priority_scale, args.refresh_hints_interval
    )

    if args.use_pretrain:
        trainer.pretrain_supervised(args.pretrain_epochs, args.pretrain_lr)

    if args.replay_warmup:
        trainer.warmup_replay_with_expert()

    trainer.train(args.num_episodes)

    test_metrics = trainer.evaluate(test_tcm_data, dataset_name="测试集")
    logger.info("========== 最终测试集指标汇总 ==========")
    logger.info(
        f"测试集-自主停止: sample_f1={test_metrics['auto']['sample_f1']:.4f}, "
        f"exact_match={test_metrics['auto']['exact_match']:.4f}, "
        f"micro_f1={test_metrics['auto']['micro_f1']:.4f}, "
        f"macro_f1={test_metrics['auto']['macro_f1']:.4f}, "
        f"supported_macro_f1={test_metrics['auto'].get('supported_macro_f1', 0.0):.4f}, "
        f"勿选率={test_metrics['auto'].get('false_selection_rate', 0.0):.4f}, "
        f"漏选率={test_metrics['auto'].get('miss_selection_rate', 0.0):.4f}"
    )

    training_config = {
        'gamma': args.gamma,
        'tau': args.tau,
        'target_update_strategy': args.target_update_strategy,
        'target_update_interval': args.target_update_interval,
        'use_per': bool(args.use_per),
        'per_alpha': args.per_alpha,
        'per_beta_start': args.per_beta_start,
        'per_beta_frames': args.per_beta_frames,
        'priority_clip': args.priority_clip,
        'optimize_interval': args.optimize_interval,
        'n_step': args.n_step,
        'aux_supervised_start': args.aux_supervised_start,
        'aux_supervised_weight': args.aux_supervised_weight,
        'aux_pos_weight_max': args.aux_pos_weight_max,
        'pretrain_pos_weight_max': args.pretrain_pos_weight_max,
        'pretrain_anchor_weight': args.pretrain_anchor_weight,
        'pretrain_all_permutations': args.pretrain_all_permutations,
        'use_curriculum': bool(args.use_curriculum),
        'curriculum_stage1': args.curriculum_stage1,
        'curriculum_stage2': args.curriculum_stage2,
        'min_actions_before_stop': args.min_actions_before_stop,
        'stop_margin_threshold': args.stop_margin_threshold,
        'rare_priority_scale': args.rare_priority_scale,
        'pretrain_miss_priority_scale': args.pretrain_miss_priority_scale,
        'Se_weight_min': args.Se_weight_min,
        'Se_weight_max': args.Se_weight_max,
        'Se_weight_power': args.Se_weight_power,
        'Se_weight_method': args.Se_weight_method,
        'Se_weight_log_scale': args.Se_weight_log_scale,
    }
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = save_checkpoint(
        model_dir, policy_net, symptoms, Se, args.nn_units, args.nn_units2,
        args.dropout, state_vector_len, n_actions, args.seed, args.test_ratio, test_metrics,
        args.model_type, reward_config=env.reward_config, training_config=training_config
    )
    print(f"模型已保存至: {model_path}")
    # print(f"测试集自主停止sample_f1={test_metrics['auto']['sample_f1']:.4f}")


if __name__ == "__main__":
    main()
