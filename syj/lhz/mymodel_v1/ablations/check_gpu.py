# -*- coding: utf-8 -*-
"""点击 VS Code 右上角运行，检查当前 Python 是否能使用 NVIDIA GPU。"""

import torch


def main():
    print('=' * 64)
    print('mymodel_v1 GPU 环境检查')
    print(f'PyTorch 版本: {torch.__version__}')
    print(f'PyTorch CUDA 版本: {torch.version.cuda}')
    print(f'CUDA 是否可用: {torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        print('\n当前会使用 CPU。')
        print('Windows GPU 加速需要：NVIDIA 显卡、可用的 NVIDIA 驱动，以及 CUDA 版 PyTorch。')
        print('还要确认 VS Code 右下角选择的是安装了 CUDA 版 PyTorch 的 Python 解释器。')
        return

    print(f'检测到 GPU 数量: {torch.cuda.device_count()}')
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gb = properties.total_memory / (1024 ** 3)
        print(
            f'GPU {index}: {properties.name}, 显存={memory_gb:.1f} GB, '
            f'计算能力={properties.major}.{properties.minor}'
        )
    device = torch.device('cuda:0')
    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    result = left @ right
    torch.cuda.synchronize(device)
    print(f'\nCUDA 矩阵计算成功: shape={tuple(result.shape)}')
    print('消融实验将自动使用 CUDA，并启用 AMP；可以直接运行各子目录的 run.py。')


if __name__ == '__main__':
    main()
