# -*- coding: utf-8 -*-
"""V1 数据读取、症状组划分和训练集词表拟合。

数据集不含患者/病例 ID，V1 以规范化症状集合为泄漏代理组：同一症状组的所有
记录只允许出现在 train、validation、test 中的一个集合。该规则防止重复症状模板
跨集合，但不等同于患者级独立划分，相关限制会写入 split manifest。
"""

import csv
import hashlib
import json
import math
import os
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_PROTOCOL_VERSION = 'symptom_group_v1'


def normalize_tokens(tokens):
    """统一 Unicode、去空白/重复并排序，避免相同症状因顺序不同跨集合。"""
    return tuple(sorted({
        unicodedata.normalize('NFKC', str(token)).strip()
        for token in tokens if str(token).strip()
    }))


def symptom_group_key(symptoms):
    return '|'.join(normalize_tokens(symptoms))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_tcm_data(filename, logger=None):
    """解析三列 txt；此阶段绝不创建词表、类别权重或 Top-N 标签集合。"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f'未找到数据集：{filename}')

    records, skipped = [], 0
    with open(filename, 'r', encoding='utf-8') as file_obj:
        header = file_obj.readline().strip()
        if logger is not None:
            logger.info(f'数据集表头: {header}')
        for line_no, line in enumerate(file_obj, start=2):
            parts = line.strip().split(maxsplit=2)
            if len(parts) != 3:
                skipped += 1
                continue
            symptoms = normalize_tokens(parts[1].split(','))
            labels = normalize_tokens(parts[2].split(','))
            if not symptoms or not labels:
                skipped += 1
                continue
            records.append({
                'record_id': parts[0],
                'line_no': line_no,
                'symptoms': symptoms,
                'labels': labels,
                'group_key': symptom_group_key(symptoms),
            })
    if not records:
        raise ValueError(f'数据集为空或不含有效三列记录：{filename}')
    if logger is not None:
        logger.info(
            f'原始有效记录:{len(records)}, 症状组:{len({r["group_key"] for r in records})}, '
            f'跳过无效记录:{skipped}, 划分协议:{SPLIT_PROTOCOL_VERSION}'
        )
    return records


def _target_counts(total, ratios):
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for idx in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[:total - sum(counts)]:
        counts[idx] += 1
    return counts


def group_split_data(records, seed, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, logger=None):
    """症状组互斥的三路划分，并先为每个标签保留至少一个训练组。

    先以贪心 set-cover 选择少量训练锚点组，防止稀有标签被全部放入验证/测试；
    再按目标样本量将其余组分配到三路，避免旧实现把每个含新标签的组都强制放入训练。
    """
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError('train/val/test 比例必须均大于0且总和为1')

    grouped = defaultdict(list)
    for record in records:
        grouped[record['group_key']].append(record)
    label_group_frequency = Counter()
    groups = []
    for key, group_records in grouped.items():
        labels = set().union(*(set(record['labels']) for record in group_records))
        label_group_frequency.update(labels)
        groups.append({'key': key, 'records': group_records, 'labels': labels, 'size': len(group_records)})

    rng = random.Random(seed)
    rng.shuffle(groups)
    targets = _target_counts(len(records), ratios)
    bucket_groups = [[], [], []]
    bucket_sizes = [0, 0, 0]
    bucket_labels = [Counter(), Counter(), Counter()]

    # 仅选覆盖全部标签所必需的锚点组进入 train；不把所有含“新标签”的组都放入 train。
    uncovered = set(label_group_frequency)
    remaining = list(groups)
    while uncovered:
        best = max(
            remaining,
            key=lambda group: (
                sum(1.0 / label_group_frequency[label] for label in group['labels'] & uncovered),
                len(group['labels'] & uncovered),
                -group['size'],
            ),
        )
        bucket_groups[0].append(best)
        bucket_sizes[0] += best['size']
        bucket_labels[0].update(best['labels'])
        uncovered -= best['labels']
        remaining.remove(best)

    # 其余组优先填补最欠缺的集合；标签项只用于减小明显分布偏差，不再破坏比例。
    for group in remaining:
        best_index, best_score = None, None
        for index in (0, 1, 2):
            projected = bucket_sizes[index] + group['size']
            # 使用“填充比例”而非距目标绝对误差：否则分母较大的 train 会持续获选。
            fill_ratio = projected / max(targets[index], 1)
            overflow = max(0, projected - targets[index]) / max(targets[index], 1)
            missing_label_bonus = sum(
                1.0 / max(label_group_frequency[label], 1)
                for label in group['labels'] if bucket_labels[index][label] == 0
            )
            score = fill_ratio + 3.0 * overflow - 0.05 * missing_label_bonus
            if best_score is None or score < best_score:
                best_index, best_score = index, score
        bucket_groups[best_index].append(group)
        bucket_sizes[best_index] += group['size']
        bucket_labels[best_index].update(group['labels'])

    split_names = ('train', 'validation', 'test')
    split_records = {}
    for name, groups_for_split in zip(split_names, bucket_groups):
        records_for_split = [record for group in groups_for_split for record in group['records']]
        rng.shuffle(records_for_split)
        split_records[name] = records_for_split
    validate_group_disjoint(split_records)
    if logger is not None:
        logger.info('症状组划分完成: ' + ', '.join(
            f'{name}={len(split_records[name])} records/{len(bucket_groups[idx])} groups'
            for idx, name in enumerate(split_names)
        ))
    return split_records


def validate_group_disjoint(split_records):
    names = tuple(split_records)
    group_sets = {name: {record['group_key'] for record in split_records[name]} for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise AssertionError(f'症状组泄漏：{left} 和 {right} 共享 {len(overlap)} 个组')
    return group_sets


def fit_training_schema(train_records, max_se_num=0, logger=None):
    """仅用训练集拟合输入词表、标签空间、标签 support 和 Top-N 过滤规则。"""
    symptom_counts = Counter(symptom for record in train_records for symptom in record['symptoms'])
    label_counts = Counter(label for record in train_records for label in record['labels'])
    if not label_counts:
        raise ValueError('训练集不含可学习标签')
    ordered_labels = sorted(label_counts, key=lambda label: (-label_counts[label], label))
    if max_se_num and 0 < max_se_num < len(ordered_labels):
        selected_labels = tuple(ordered_labels[:max_se_num])
    else:
        selected_labels = tuple(ordered_labels)
    schema = {
        # <UNK_SYM> 只收集验证/测试未见症状，不改变已知症状索引；训练中该维度为0。
        'symptoms': {idx: symptom for idx, symptom in enumerate(sorted(symptom_counts))},
        'labels': {idx: label for idx, label in enumerate(selected_labels)},
        'selected_labels': set(selected_labels),
        'train_label_support': {label: int(label_counts[label]) for label in selected_labels},
        'max_se_num': int(max_se_num),
    }
    if logger is not None:
        logger.info(f'训练集词表: 症状数={len(schema["symptoms"])}, 标签数={len(schema["labels"])}')
        logger.info(f'训练标签支持度: {schema["train_label_support"]}')
    return schema


def transform_split(records, schema, split_name, unknown_label_policy='drop_sample'):
    """应用训练标签空间；症状 OOV 保留并在编码时忽略，同时记录统计。"""
    known_symptoms = set(schema['symptoms'].values())
    known_labels = schema['selected_labels']
    output, stats = [], Counter()
    for record in records:
        unknown_labels = tuple(label for label in record['labels'] if label not in known_labels)
        if unknown_labels:
            stats['samples_with_unknown_labels'] += 1
            stats['unknown_label_occurrences'] += len(unknown_labels)
            stats['unknown_label_types'] += len(set(unknown_labels))
            if unknown_label_policy == 'drop_sample':
                stats['dropped_unknown_label_samples'] += 1
                continue
            raise ValueError(f'{split_name} 出现训练未见标签：{unknown_labels}')
        unknown_symptoms = tuple(symptom for symptom in record['symptoms'] if symptom not in known_symptoms)
        if unknown_symptoms:
            stats['samples_with_oov_symptoms'] += 1
            stats['oov_symptom_occurrences'] += len(unknown_symptoms)
        output.append((list(record['symptoms']), list(record['labels']), record['record_id'], record['group_key']))
    stats['input_records'] = len(records)
    stats['output_records'] = len(output)
    return output, dict(stats)


def export_split_data(split_data, output_dir, manifest, logger=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in split_data.items():
        with (output_dir / f'{split_name}_data.csv').open('w', encoding='utf-8-sig', newline='') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(['split_index', 'record_id', 'symptom_group_key', 'symptom', 'Se'])
            for index, item in enumerate(records, start=1):
                writer.writerow([index, item[2], item[3], ','.join(item[0]), ','.join(item[1])])
    manifest_path = output_dir / 'split_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if logger is not None:
        logger.info(f'已导出 train/validation/test 与 split manifest: {output_dir}')
    return manifest_path


def build_split_manifest(data_path, seed, ratios, raw_records, split_records, schema, transform_stats):
    group_sets = validate_group_disjoint(split_records)
    return {
        'protocol_version': SPLIT_PROTOCOL_VERSION,
        'limitation': '症状组互斥划分不是患者级独立划分；数据源不含患者/病例ID。',
        'data_path': str(Path(data_path).resolve()),
        'data_sha256': file_sha256(data_path),
        'split_seed': int(seed),
        'ratios': {'train': ratios[0], 'validation': ratios[1], 'test': ratios[2]},
        'raw_record_count': len(raw_records),
        'split_record_counts': {name: len(records) for name, records in split_records.items()},
        'split_group_counts': {name: len(group_sets[name]) for name in split_records},
        'training_vocabulary': {
            'symptom_count': len(schema['symptoms']),
            'label_count': len(schema['labels']),
            'labels': list(schema['labels'].values()),
            'label_support': schema['train_label_support'],
        },
        'transform_stats': transform_stats,
    }


def strip_sample_id(data):
    return [(item[0], item[1]) for item in data]


def compute_Se_weights(training_data, env, logger=None, min_weight=0.75, max_weight=2.0, power=0.5):
    """固定训练集频次权重；均值归一以避免整体奖励尺度随标签数漂移。"""
    freq = Counter(label for _, labels in training_data for label in labels)
    total = sum(freq.values())
    weights = {}
    for action_idx in range(env.Se_action_num):
        label = env.action_space[action_idx]
        count = max(freq.get(label, 1), 1)
        raw = (total / (max(env.Se_action_num, 1) * count)) ** power if total else 1.0
        weights[action_idx] = float(min(max(raw, min_weight), max_weight))
    mean_weight = sum(weights.values()) / max(len(weights), 1)
    weights = {idx: value / mean_weight for idx, value in weights.items()}
    if logger is not None:
        logger.info(f'训练集标签权重: { {env.action_space[idx]: round(value, 4) for idx, value in weights.items()} }')
    return weights


def compute_Se_supports(training_data, env, logger=None):
    freq = Counter(label for _, labels in training_data for label in labels)
    supports = {idx: int(freq.get(env.action_space[idx], 0)) for idx in range(env.Se_action_num)}
    if logger is not None:
        logger.info(f'训练集标签support: { {env.action_space[idx]: supports[idx] for idx in supports} }')
    return supports
