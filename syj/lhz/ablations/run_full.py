# -*- coding: utf-8 -*-
"""完整模型消融对照：监督预训练 + 专家轨迹预填充 + 后续RL训练。"""

from ablation_utils import wrapper_cli


if __name__ == '__main__':
    wrapper_cli('full')
