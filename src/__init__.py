"""
TCM Syndrome Differentiation using Reinforcement Learning (DQN).
"""

from __future__ import absolute_import  # 兼容性导入，确保在不同版本的Python中使用绝对导入

__version__ = "0.1"  # 定义脚本的版本号
__description__ = "TCM Syndrome Differentiation using Reinforcement Learning (DQN)." # 定义脚本的描述

# Constants for the model and training
STATE_SIZE = 100  # Adjust based on the number of symptoms 状态的维度，通常根据症状的数量进行调整
ACTION_SIZE = 50  # Adjust based on the number of state elements 动作的维度，通常根据状态要素的数量进行调整

EPISODES = 400  # 训练的总轮数
BATCH_SIZE = 32  # 每个训练批次的样本数量
