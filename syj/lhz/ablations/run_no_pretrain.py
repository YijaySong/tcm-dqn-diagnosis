# -*- coding: utf-8 -*-
"""无预训练消融：关闭监督预训练，保留专家轨迹预填充和后续RL训练。"""

from ablation_utils import wrapper_cli


if __name__ == '__main__':
    wrapper_cli('no_pretrain')
