# -*- coding: utf-8 -*-
"""通用工具文件。

提供训练日志创建、随机种子固定、项目根目录定位和默认数据集路径等辅助功能，
供主程序和训练流程复用。
"""

import logging
import os
import random

import numpy as np
import torch


def create_logger(logname):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(logname, mode='w+')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_accelerator(selected_device, enable_tf32=True):
    """配置不改变模型结构的 CUDA 加速，并返回可写入日志的设备信息。"""
    info = {
        'device': str(selected_device),
        'cuda_available': bool(torch.cuda.is_available()),
        'gpu_name': None,
        'cuda_version': torch.version.cuda,
        'tf32': False,
    }
    if selected_device.type != 'cuda':
        return info
    index = selected_device.index if selected_device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    info['gpu_name'] = torch.cuda.get_device_name(index)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high' if enable_tf32 else 'highest')
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    info['tf32'] = bool(enable_tf32)
    return info


def capture_rng_state():
    """保存所有训练随机源，供 ``last.pt`` 恢复训练使用。"""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """恢复 :func:`capture_rng_state` 保存的状态。"""
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def default_data_path():
    return os.path.join(project_root(), 'dataset', 'lhz_data_cleaned.txt')
