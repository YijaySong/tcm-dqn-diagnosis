# -*- coding: utf-8 -*-
"""数据处理文件。

负责读取dataset/lhz_data.txt，构建刻下症和证候要素映射，划分训练集/测试集，
导出本次划分结果，并根据训练集标签频次计算证候要素奖励权重。
"""

import csv
import glob
import math
import os
import random
from collections import Counter

try:
    import importlib
    sklearn_model_selection = importlib.import_module('sklearn.model_selection')
    train_test_split = sklearn_model_selection.train_test_split
except ImportError:
    train_test_split = None


def get_tcm_data(filename, max_Se_num=0, logger=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"未找到数据集文件：{filename}")

    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        header = file.readline().strip()
        if logger is not None:
            logger.info(f"数据集表头: {header}")
        for line_no, line in enumerate(file, start=2):
            line = line.strip()
            if line:
                data.append((line_no, line))

    tuples4gen = []
    symptom_set = set()
    Se_set = set()
    symptom_map = {}
    Se_map = {}
    Se_freq_map = {}
    combo_freq_map = Counter()
    skipped_empty_column = 0

    for line_no, piece in data:
        parts = piece.split(maxsplit=2)
        if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
            skipped_empty_column += 1
            if logger is not None:
                logger.warning(f"跳过存在空列的数据行 {line_no}: {piece}")
            continue

        symptoms = [symptom.strip() for symptom in parts[1].split(',') if symptom.strip()]
        Se_list = [Se.strip() for Se in parts[2].split(',') if Se.strip()]
        if not symptoms or not Se_list:
            skipped_empty_column += 1
            if logger is not None:
                logger.warning(f"跳过症状或Se为空的数据行 {line_no}: {piece}")
            continue
        combo_freq_map[tuple(sorted(Se_list))] += 1

        for Se in Se_list:
            if Se not in Se_set:
                Se_map[len(Se_set)] = Se
                Se_set.add(Se)
                Se_freq_map[Se] = 1
            else:
                Se_freq_map[Se] += 1

        for symp in symptoms:
            if symp not in symptom_set:
                symptom_map[len(symptom_set)] = symp
                symptom_set.add(symp)
        tuples4gen.append((symptoms, Se_list, parts[0]))

    if logger is not None:
        logger.info(f"全量刻下症数: {len(symptom_map)}, 全量证候要素数: {len(Se_map)}")
        if skipped_empty_column:
            logger.warning(f"跳过存在空列的数据行: {skipped_empty_column}条")
    if not tuples4gen:
        raise ValueError(f"数据集为空或没有有效的三列数据：{filename}")
    if logger is not None:
        logger.info(f"证候要素频次: {dict(sorted(Se_freq_map.items(), key=lambda x: (-x[1], x[0])))}")
        logger.info(f"证候要素组合数: {len(combo_freq_map)}")

    if max_Se_num and 0 < max_Se_num < len(Se_map):
        sorted_Se = sorted(Se_freq_map.items(), key=lambda x: (-x[1], x[0]))
        top_n_Se = {Se for Se, _ in sorted_Se[:max_Se_num]}
        if logger is not None:
            logger.info(f"选择频率最高的前{max_Se_num}个证候要素: {list(top_n_Se)}")
    else:
        top_n_Se = set(Se_freq_map.keys())
        if logger is not None:
            logger.info("保留全部证候要素")

    filt_tuples4gen = []
    filt_symptom_map = {}
    filt_Se_map = {}
    symptom_set.clear()
    Se_set.clear()

    max_Se_len = 0
    for item in tuples4gen:
        symptoms, Se_list, source_id = item
        if set(Se_list).issubset(top_n_Se):
            filt_tuples4gen.append((symptoms, Se_list, source_id))
            if len(Se_list) > max_Se_len:
                max_Se_len = len(Se_list)

            for Se in Se_list:
                if Se not in Se_set:
                    filt_Se_map[len(Se_set)] = Se
                    Se_set.add(Se)
            for symp in symptoms:
                if symp not in symptom_set:
                    filt_symptom_map[len(symptom_set)] = symp
                    symptom_set.add(symp)

    if logger is not None:
        logger.info(
            f"过滤后数据量: {len(filt_tuples4gen)}, "
            f"刻下症数: {len(filt_symptom_map)}, 证候要素数: {len(filt_Se_map)}"
        )

    return filt_tuples4gen, filt_symptom_map, filt_Se_map, max_Se_len


def stratified_split_data(tcm_data, seed, test_ratio=0.2, logger=None):
    rng = random.Random(seed)
    combo_keys = ['|'.join(sorted(item[1])) for item in tcm_data]
    combo_counts = Counter(combo_keys)
    if logger is not None:
        logger.info(f"数据集划分比例: 训练集 {1 - test_ratio:.0%}, 测试集 {test_ratio:.0%}")

    if train_test_split is not None and combo_counts and min(combo_counts.values()) >= 2:
        train_data, test_data = train_test_split(tcm_data, test_size=test_ratio, random_state=seed, stratify=combo_keys)
        if logger is not None:
            logger.info("使用证候要素组合分层划分训练集/测试集")
            logger.info(f"数据集总量:{len(tcm_data)}, 训练集:{len(train_data)}, 测试集:{len(test_data)}")
        return list(train_data), list(test_data)

    if logger is not None:
        if train_test_split is None:
            logger.warning("当前Python环境未安装/未识别scikit-learn，使用内置分层划分逻辑")
        elif not combo_counts or min(combo_counts.values()) < 2:
            logger.warning("存在样本数不足2的证候要素组合，使用内置随机划分逻辑")

    grouped = {}
    for item, key in zip(tcm_data, combo_keys):
        grouped.setdefault(key, []).append(item)

    train_data = []
    test_data = []
    for items in grouped.values():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_ratio))) if len(items) >= 2 else 0
        n_test = min(n_test, len(items) - 1) if len(items) >= 2 else 0
        test_data.extend(items[:n_test])
        train_data.extend(items[n_test:])

    if not test_data:
        shuffled = list(tcm_data)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_data = shuffled[:n_test]
        train_data = shuffled[n_test:]

    rng.shuffle(train_data)
    rng.shuffle(test_data)
    if logger is not None:
        logger.info("使用内置训练集/测试集划分")
        logger.info(f"数据集总量:{len(tcm_data)}, 训练集:{len(train_data)}, 测试集:{len(test_data)}")
    return train_data, test_data


def strip_sample_id(data):
    """去掉导出用的原始样本编号，只保留训练需要的(症状列表, 证候要素列表)。"""
    stripped = []
    for item in data:
        symptoms, Se_names = item[0], item[1]
        stripped.append((symptoms, Se_names))
    return stripped


def export_split_data(train_data, test_data, output_dir, seed, test_ratio, logger=None):
    """导出本次训练/测试划分，方便人工查看每条样本属于哪个集合。"""
    os.makedirs(output_dir, exist_ok=True)
    for old_path in glob.glob(os.path.join(output_dir, 'train_data_seed*_test*.csv')):
        os.remove(old_path)
    for old_path in glob.glob(os.path.join(output_dir, 'test_data_seed*_test*.csv')):
        os.remove(old_path)

    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")

    def write_csv(path, data):
        with open(path, 'w', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['split_index', 'source_id', 'symptom', 'Se'])
            for split_index, item in enumerate(data, start=1):
                symptoms = item[0]
                Se_names = item[1]
                source_id = item[2] if len(item) >= 3 else ''
                writer.writerow([
                    split_index,
                    source_id,
                    ','.join(symptoms),
                    ','.join(Se_names),
                ])

    write_csv(train_path, train_data)
    write_csv(test_path, test_data)

    if logger is not None:
        logger.info(f"训练集明细已导出并覆盖: {train_path}")
        logger.info(f"测试集明细已导出并覆盖: {test_path}")
        logger.info(f"本次划分参数: seed={seed}, test_ratio={test_ratio:.2f}")
    return train_path, test_path


def compute_Se_weights(training_data, env, logger=None, min_weight=0.75, max_weight=2.5, power=0.5):
    freq = Counter()
    for _, Se_names in training_data:
        for Se_name in Se_names:
            freq[env.swapped_action_space[Se_name]] += 1
    total = sum(freq.values())
    weights = {}
    for action_idx in range(env.Se_action_num):
        count = max(freq.get(action_idx, 1), 1)
        weight = (total / (env.Se_action_num * count)) ** power
        weights[action_idx] = float(min(max(weight, min_weight), max_weight))
    if logger is not None:
        logger.info(
            f"证候要素奖励权重(min={min_weight}, max={max_weight}, power={power}): "
            f"{ {env.action_space[k]: round(v, 4) for k, v in weights.items()} }"
        )
    return weights
