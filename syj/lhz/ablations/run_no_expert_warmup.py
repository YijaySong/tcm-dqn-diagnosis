# -*- coding: utf-8 -*-
"""无专家轨迹预填充消融：保留监督预训练和后续RL训练，关闭ReplayMemory专家预填充。"""

from ablation_utils import wrapper_cli


if __name__ == '__main__':
    wrapper_cli('no_expert_warmup')
