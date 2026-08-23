# -*- coding: utf-8 -*-
"""仅预训练消融：只做监督预训练，不做专家轨迹预填充和后续RL训练。"""

from ablation_utils import wrapper_cli


if __name__ == '__main__':
    wrapper_cli('pretrain_only')
