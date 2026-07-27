# -*- coding: utf-8 -*-
"""常量配置文件。

定义整个DQN辨证项目共用的常量和计算设备，例如停止动作名称、负无穷mask值、
以及自动选择CPU/CUDA/MPS设备。
"""

import torch


STOP_ACTION = "停止"
# 合法动作掩码会在 CUDA AMP 下写入 FP16 Q 值；-inf 对所有浮点 dtype 都可表示。
NEG_INF = float('-inf')

def resolve_device(request='auto'):
    """解析训练设备；Windows + NVIDIA 优先使用 CUDA。"""
    name = str(request or 'auto').lower()
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    selected = torch.device(name)
    if selected.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                '请求使用 CUDA，但当前 PyTorch 检测不到 NVIDIA GPU。'
                '请安装 NVIDIA 驱动和 CUDA 版 PyTorch。'
            )
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f'请求 cuda:{index}，但只检测到 {torch.cuda.device_count()} 张 CUDA GPU')
    if selected.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('请求使用 MPS，但当前环境不支持 MPS')
    return selected


device = resolve_device('auto')
