# -*- coding: utf-8 -*-
"""常量配置文件。

定义整个DQN辨证项目共用的常量和计算设备，例如停止动作名称、负无穷mask值、
以及自动选择CPU/CUDA/MPS设备。
"""

import torch


STOP_ACTION = "停止"
NEG_INF = -1e9

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
