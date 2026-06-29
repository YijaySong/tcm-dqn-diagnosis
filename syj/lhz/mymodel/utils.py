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


def project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def default_data_path():
    return os.path.join(project_root(), 'dataset', 'lhz_data.txt')
