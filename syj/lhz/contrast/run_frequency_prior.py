# -*- coding: utf-8 -*-
"""频率先验基线模型。"""

from argparse import ArgumentParser

import numpy as np

from common import add_common_arguments, run_score_model

MODEL_SPEC = {
    'name': 'frequency_prior',
    'label': 'Frequency Prior',
    'description': '按训练集证候要素频率进行固定排序预测的下限基线。',
    'category': 'baseline',
}


def build_scores(x_train, y_train, x_test, dataset, logger):
    label_freq = y_train.mean(axis=0)
    if label_freq.max() > 0:
        scores = label_freq / label_freq.max()
    else:
        scores = np.ones(y_train.shape[1], dtype=float)
    logger.info(f'训练集标签平均阳性率: {float(label_freq.mean()):.6f}')
    return np.tile(scores.reshape(1, -1), (x_test.shape[0], 1))


def main():
    parser = add_common_arguments(ArgumentParser(description='LHZ Frequency Prior baseline'))
    parser.set_defaults(threshold=0.5)
    args = parser.parse_args()
    run_score_model(MODEL_SPEC, args, build_scores)


if __name__ == '__main__':
    main()
