# -*- coding: utf-8 -*-
"""主模型最终配置的唯一来源。

每完成一组正式消融，必须同时更新 FINAL_CONFIG、DECISION_STATUS 和对应 README。
main.py 直接读取本文件，因此点击运行时始终采用目前已经确认的最佳组合。

当前数值由清洗数据协议下的六组正式消融确定。
"""


FINAL_CONFIG_VERSION = '2026-07-20-v8-cleaned-ablation-final'


FINAL_CONFIG = {
    # 网络与探索：20260719_142310 选择普通 Dueling 网络且不加显式探索。
    'nn_units': 256,
    'nn_units2': 128,
    'dropout': 0.15,
    'model_type': 'set_dueling',
    'exploration_mode': 'none',
    'eps_start': 0.0,
    'eps_end': 0.0,
    'eps_decay': 10000,

    # 奖励函数：20260718_215858 选择训练集频率加权 F1，不加固定步成本。
    'gamma': 0.95,
    'reward_beta': 1.0,
    'step_cost': 0.0,
    'stop_utility_scale': 1.0,
    'weight_min': 0.75,
    'weight_max': 2.0,
    'weight_power': 0.5,

    # STOP：20260718_221141 选择最少输出1个标签；不增加 stop margin。
    'min_actions_before_stop': 1,
    'stop_margin_threshold': None,

    # 监督、示范和 RL：20260719_101633 选择完整混合模型。
    'use_pretrain': 0,
    'pretrain_epochs': 10,
    'pretrain_lr': 1e-3,
    'pretrain_num_orders': 1,
    'use_expert_warmup': 1,
    'use_aux_bce': 1,
    'aux_bce_weight': 0.05,
    'run_rl': 1,

    # 监督损失：20260720_094527 选择类别平衡 BCE。focal_gamma 仅供备选实验使用。
    'supervised_loss': 'class_balanced',
    'focal_gamma': 2.0,

    # Replay 长尾：20260720_094515 确认不叠加稀有度 multiplier。
    # 标准 TD-error PER 仍保留；本实验没有消融 use_per 本身。
    'use_per': 1,
    'rare_per_scale': 0.0,
}


DECISION_STATUS = {
    'reward_simplification': 'confirmed_cleaned_data_20260718_215858',
    'network_exploration': 'confirmed_cleaned_data_20260719_142310',
    'stop_policy': 'confirmed_cleaned_data_20260718_221141',
    'supervision_components': 'confirmed_cleaned_data_20260719_101633',
    'tail_supervised_loss': 'confirmed_cleaned_data_20260720_094527_one_focal_seed_failed',
    'tail_rare_per': 'confirmed_cleaned_data_20260720_094515',
}
