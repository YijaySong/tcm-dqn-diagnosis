# -*- coding: utf-8 -*-
"""纯RL消融：无监督预训练、无专家轨迹预填充，仅从随机网络开始后续RL训练。"""

from ablation_utils import wrapper_cli


if __name__ == '__main__':
    wrapper_cli('rl_only')
