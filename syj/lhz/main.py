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
from data import compute_Se_weights, export_split_data, get_tcm_data, stratified_split_data, strip_sample_id
from env import Environment
from memory import ReplayMemory
from model import DQN
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
    parser.add_argument("-lr", type=float, dest="lr", default=0.001)
    parser.add_argument("-weight_decay", type=float, dest="weight_decay", default=1e-4)
    parser.add_argument("-nn_units", type=int, dest="nn_units", default=128)
    parser.add_argument("-nn_units2", type=int, dest="nn_units2", default=64)
    parser.add_argument("-dropout", type=float, dest="dropout", default=0.1)
    parser.add_argument("-seed", type=int, dest="seed", default=9)
    parser.add_argument("-max_se_num", type=int, dest="max_Se_num", default=0)
    parser.add_argument("-use_pretrain", type=int, dest="use_pretrain", default=1)
    parser.add_argument("-pretrain_epochs", type=int, dest="pretrain_epochs", default=20)
    parser.add_argument("-pretrain_lr", type=float, dest="pretrain_lr", default=0.001)
    parser.add_argument("-replay_warmup", type=int, dest="replay_warmup", default=1)
    parser.add_argument("-test_ratio", type=float, dest="test_ratio", default=0.2)
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
    env.set_Se_weights(compute_Se_weights(training_tcm_data, env, logger=logger))

    logger.info(f"训练数据数量:{len(training_tcm_data)}, 测试数据数量:{len(test_tcm_data)}, 测试集比例:{args.test_ratio:.2f}")
    logger.info(f"最大真实证候要素数:{max_Se_len}")

    state_vector_len = len(env.state_space)
    n_actions = len(env.action_space)
    logger.info(f"状态向量长度:{state_vector_len}, 动作数:{n_actions}")

    policy_net = DQN(state_vector_len, n_actions, args.nn_units, args.nn_units2, args.dropout).to(device)
    target_net = DQN(state_vector_len, n_actions, args.nn_units, args.nn_units2, args.dropout).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ReplayMemory(args.mem_capacity)
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    logger.info(
        f"**设置**: batch={args.batch_size}, mem={args.mem_capacity}, gamma={args.gamma}, "
        f"eps={args.eps_start}->{args.eps_end}, eps_decay={args.eps_decay}, tau={args.tau}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, hidden={args.nn_units}/{args.nn_units2}, "
        f"dropout={args.dropout}, pretrain={args.use_pretrain}, warmup={args.replay_warmup}, "
        f"test_ratio={args.test_ratio}"
    )

    trainer = DQNTrainer(
        env, training_tcm_data, test_tcm_data, policy_net, target_net, memory,
        optimizer, device, logger, args.batch_size, args.gamma, args.tau,
        args.weight_decay, args.eps_start, args.eps_end, args.eps_decay, n_actions
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
        f"macro_f1={test_metrics['auto']['macro_f1']:.4f}"
    )
    logger.info(
        f"测试集-Top-2: sample_f1={test_metrics['top2']['sample_f1']:.4f}, "
        f"exact_match={test_metrics['top2']['exact_match']:.4f}, "
        f"micro_f1={test_metrics['top2']['micro_f1']:.4f}, "
        f"macro_f1={test_metrics['top2']['macro_f1']:.4f}"
    )

    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = save_checkpoint(
        model_dir, policy_net, symptoms, Se, args.nn_units, args.nn_units2,
        args.dropout, state_vector_len, n_actions, args.seed, args.test_ratio, test_metrics
    )
    print(f"模型已保存至: {model_path}")
    # print(f"测试集自主停止sample_f1={test_metrics['auto']['sample_f1']:.4f}")

    # symptoms_str = '胸闷,胸痛,畏寒,纳呆,睡后易醒,大便艰难,舌淡,舌边齿痕,舌苔白,舌苔薄,细脉,弱脉'
    # predicted_Se = trainer.predict_symptoms(symptoms_str)
    # predicted_Se_top2 = trainer.predict_symptoms(symptoms_str, force_top_k=2, max_actions=2)
    # print(f'推荐证候要素(自主停止): {",".join(predicted_Se) if predicted_Se else "无"}')
    # print(f'推荐证候要素(固定Top-2诊断): {",".join(predicted_Se_top2) if predicted_Se_top2 else "无"}')


if __name__ == "__main__":
    main()
