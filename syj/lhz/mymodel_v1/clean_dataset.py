# -*- coding: utf-8 -*-
"""生成可审计的 LHZ 清洗数据集；直接点击运行即可重复生成全部产物。"""

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / 'dataset'
DEFAULT_SOURCE = DATASET_DIR / 'lhz_data.txt'
DEFAULT_OUTPUT = DATASET_DIR / 'lhz_data_cleaned.txt'
OUTLIER_PERCENTILE = 0.95
MIN_LABEL_GROUP_SUPPORT = 5

# 这两类 token 是数据导出时混入的时长/术后字段，不是刻下症。
NON_SYMPTOM_PATTERNS = (
    re.compile(r'^\d+(?:天|周|月|年)$'),
    re.compile(r'^介入术后\d+$'),
)
INVALID_LABELS = {'乏力'}
LABEL_ALIASES = {'血瘀证': '血瘀'}


def normalize_tokens(tokens):
    return tuple(sorted({
        unicodedata.normalize('NFKC', token).strip()
        for token in tokens if token.strip()
    }))


def is_non_symptom_metadata(token):
    return any(pattern.fullmatch(token) for pattern in NON_SYMPTOM_PATTERNS)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank_percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('无法从空数据计算百分位数')
    index = max(0, math.ceil(float(percentile) * len(ordered)) - 1)
    return ordered[index]


def parse_source(path):
    records, malformed = [], []
    with Path(path).open('r', encoding='utf-8') as file_obj:
        header = file_obj.readline().strip()
        for line_no, line in enumerate(file_obj, start=2):
            parts = line.strip().split(maxsplit=2)
            if len(parts) != 3:
                malformed.append({'line_no': line_no, 'raw_line': line.rstrip('\n')})
                continue
            record_id, symptom_text, label_text = parts
            original_symptoms = normalize_tokens(symptom_text.split(','))
            original_labels = normalize_tokens(label_text.split(','))
            removed_symptoms = tuple(token for token in original_symptoms if is_non_symptom_metadata(token))
            symptoms = tuple(token for token in original_symptoms if token not in removed_symptoms)

            removed_labels, alias_changes, cleaned_labels = [], [], []
            for label in original_labels:
                if label in INVALID_LABELS:
                    removed_labels.append(label)
                    continue
                cleaned = LABEL_ALIASES.get(label, label)
                if cleaned != label:
                    alias_changes.append(f'{label}->{cleaned}')
                cleaned_labels.append(cleaned)
            labels = normalize_tokens(cleaned_labels)
            records.append({
                'record_id': record_id,
                'line_no': line_no,
                'original_symptoms': original_symptoms,
                'original_labels': original_labels,
                'symptoms': symptoms,
                'labels': labels,
                'removed_symptoms': removed_symptoms,
                'removed_labels': tuple(removed_labels),
                'alias_changes': tuple(alias_changes),
            })
    return header, records, malformed


def record_sort_key(record):
    value = record['record_id']
    return (0, int(value)) if value.isdigit() else (1, value)


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def clean_records(records):
    audit, quarantine = [], []
    symptom_counts = [len(record['symptoms']) for record in records if record['symptoms']]
    symptom_threshold = nearest_rank_percentile(symptom_counts, OUTLIER_PERCENTILE)
    candidates = []

    for record in records:
        if record['removed_symptoms']:
            audit.append({
                'action': 'remove_non_symptom_metadata',
                'reason': ','.join(record['removed_symptoms']),
                'source_record_ids': record['record_id'],
                'source_line_numbers': str(record['line_no']),
                'symptoms': ','.join(record['symptoms']),
                'original_label_variants': ','.join(record['original_labels']),
                'cleaned_labels': ','.join(record['labels']),
            })
        if record['removed_labels'] or record['alias_changes']:
            changes = [f'删除标签:{value}' for value in record['removed_labels']]
            changes.extend(f'标签映射:{value}' for value in record['alias_changes'])
            audit.append({
                'action': 'correct_label_token',
                'reason': ';'.join(changes),
                'source_record_ids': record['record_id'],
                'source_line_numbers': str(record['line_no']),
                'symptoms': ','.join(record['symptoms']),
                'original_label_variants': ','.join(record['original_labels']),
                'cleaned_labels': ','.join(record['labels']),
            })

        reason = None
        if not record['symptoms']:
            reason = '清洗后症状为空'
        elif not record['labels']:
            reason = '清洗后标签为空'
        elif len(record['labels']) == 1 and len(record['symptoms']) > symptom_threshold:
            reason = f'单标签且症状数{len(record["symptoms"])}超过P95阈值{symptom_threshold}'
        if reason:
            quarantine.append({
                'record_id': record['record_id'],
                'line_no': record['line_no'],
                'reason': reason,
                'symptom_count': len(record['symptoms']),
                'symptoms': ','.join(record['symptoms']),
                'labels': ','.join(record['labels']),
            })
        else:
            candidates.append(record)

    grouped = defaultdict(list)
    for record in candidates:
        grouped[record['symptoms']].append(record)

    consensus_by_group, conflict_group_count = {}, 0
    duplicate_excess, ambiguous_label_removals = 0, 0
    empty_consensus_groups = 0
    for symptoms, group in grouped.items():
        group.sort(key=record_sort_key)
        variants = Counter(record['labels'] for record in group)
        duplicate_excess += sum(count - 1 for count in variants.values())
        source_ids = [record['record_id'] for record in group]
        source_lines = [record['line_no'] for record in group]

        if len(variants) > 1:
            conflict_group_count += 1
            label_sets = [set(labels) for labels in variants]
            consensus = set.intersection(*label_sets)
            union = set.union(*label_sets)
            removed_ambiguous = sorted(union - consensus)
            ambiguous_label_removals += len(removed_ambiguous)
            audit.append({
                'action': 'resolve_conflicting_labels_by_intersection',
                'reason': f'移除无共识标签:{",".join(removed_ambiguous) if removed_ambiguous else "无"}',
                'source_record_ids': ','.join(source_ids),
                'source_line_numbers': ','.join(map(str, source_lines)),
                'symptoms': ','.join(symptoms),
                'original_label_variants': _json_text({','.join(key): value for key, value in variants.items()}),
                'cleaned_labels': ','.join(sorted(consensus)),
            })
            if not consensus:
                empty_consensus_groups += 1
                for record in group:
                    quarantine.append({
                        'record_id': record['record_id'], 'line_no': record['line_no'],
                        'reason': '同症标签冲突且无共同标签', 'symptom_count': len(symptoms),
                        'symptoms': ','.join(symptoms), 'labels': ','.join(record['labels']),
                    })
                continue
            labels = tuple(sorted(consensus))
        else:
            labels = next(iter(variants))

        consensus_by_group[symptoms] = labels

    # 支持度按唯一症状组计算，避免保留的重复记录把低频标签伪装成高频标签。
    group_label_support = Counter(
        label for labels in consensus_by_group.values() for label in labels
    )
    low_support_labels = {
        label for label, support in group_label_support.items()
        if support < MIN_LABEL_GROUP_SUPPORT
    }

    cleaned, removed_empty_after_filter = [], 0
    for symptoms, group in grouped.items():
        if symptoms not in consensus_by_group:
            continue
        consensus_labels = consensus_by_group[symptoms]
        removed_labels = sorted(set(consensus_labels) & low_support_labels)
        labels = tuple(label for label in consensus_labels if label not in low_support_labels)
        if removed_labels:
            audit.append({
                'action': 'remove_low_group_support_labels',
                'reason': f'唯一症状组支持度<{MIN_LABEL_GROUP_SUPPORT}:{",".join(removed_labels)}',
                'source_record_ids': ','.join(record['record_id'] for record in group),
                'source_line_numbers': ','.join(str(record['line_no']) for record in group),
                'symptoms': ','.join(symptoms),
                'original_label_variants': ','.join(consensus_labels),
                'cleaned_labels': ','.join(labels),
            })
        if not labels:
            for record in group:
                removed_empty_after_filter += 1
                quarantine.append({
                    'record_id': record['record_id'], 'line_no': record['line_no'],
                    'reason': '删除低支持度标签后标签为空', 'symptom_count': len(symptoms),
                    'symptoms': ','.join(symptoms), 'labels': ','.join(consensus_labels),
                })
            continue

        # 按用户要求保留每一条原始重复记录，但同症组使用同一个共识标签集合。
        for record in group:
            cleaned.append({
                'record_id': record['record_id'],
                'source_record_ids': [record['record_id']],
                'symptoms': symptoms,
                'labels': labels,
            })

    cleaned.sort(key=record_sort_key)
    cleaned_by_symptoms = defaultdict(set)
    symptoms_by_label_set = defaultdict(set)
    for record in cleaned:
        cleaned_by_symptoms[record['symptoms']].add(record['labels'])
        symptoms_by_label_set[record['labels']].add(record['symptoms'])
    same_symptom_conflicts = sum(
        len(label_sets) > 1 for label_sets in cleaned_by_symptoms.values()
    )
    if same_symptom_conflicts:
        raise AssertionError(f'清洗后仍有{same_symptom_conflicts}个同症不同结果组')

    stats = {
        'single_label_symptom_threshold': symptom_threshold,
        'minimum_label_group_support': MIN_LABEL_GROUP_SUPPORT,
        'candidate_record_count': len(candidates),
        'cleaned_record_count': len(cleaned),
        'cleaned_symptom_group_count': len({record['symptoms'] for record in cleaned}),
        'removed_record_count': len(quarantine),
        'removed_empty_after_low_support_filter_count': removed_empty_after_filter,
        'conflicting_symptom_group_count': conflict_group_count,
        'empty_consensus_group_count': empty_consensus_groups,
        'exact_duplicate_excess_count_retained': duplicate_excess,
        'repeated_symptom_record_excess_retained': len(cleaned) - len({record['symptoms'] for record in cleaned}),
        'removed_low_support_label_count': len(low_support_labels),
        'removed_low_support_labels': sorted(low_support_labels),
        'ambiguous_label_type_removals_across_groups': ambiguous_label_removals,
        'same_symptom_different_result_group_count_after_cleaning': same_symptom_conflicts,
        'label_sets_shared_by_different_symptoms': sum(
            len(symptom_sets) > 1 for symptom_sets in symptoms_by_label_set.values()
        ),
        'different_symptom_groups_in_shared_label_sets': sum(
            len(symptom_sets) for symptom_sets in symptoms_by_label_set.values()
            if len(symptom_sets) > 1
        ),
    }
    return cleaned, audit, quarantine, stats, group_label_support, low_support_labels


def write_cleaned_data(path, cleaned):
    with Path(path).open('w', encoding='utf-8', newline='') as file_obj:
        file_obj.write('id symptom Se\n')
        for record in cleaned:
            file_obj.write(
                f'{record["record_id"]} {",".join(record["symptoms"])} {",".join(record["labels"])}\n'
            )


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_label_review(records, cleaned, group_label_support, low_support_labels):
    raw_support = Counter(label for record in records for label in record['labels'])
    clean_support = Counter(label for record in cleaned for label in record['labels'])
    labels = sorted(set(raw_support) | set(clean_support))
    rows = []
    for label in labels:
        if label in INVALID_LABELS:
            status, note = 'removed', '症状词误入标签列'
        elif label in LABEL_ALIASES:
            status, note = 'mapped', f'映射为{LABEL_ALIASES[label]}'
        elif label in low_support_labels:
            status, note = 'removed_low_support', f'唯一症状组支持度<{MIN_LABEL_GROUP_SUPPORT}'
        elif clean_support[label] == 0:
            status, note = 'removed_by_consensus', '仅出现在无共识标注中'
        else:
            status, note = 'kept', ''
        rows.append({
            'label': label,
            'raw_support': raw_support[label],
            'unique_group_support': group_label_support.get(label, 0),
            'cleaned_support': clean_support[label],
            'status': status,
            'note': note,
        })
    return rows


def run_cleaning(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('清洗输出不能覆盖原始数据集')
    if not source.exists():
        raise FileNotFoundError(f'未找到原始数据集: {source}')
    output.parent.mkdir(parents=True, exist_ok=True)

    header, records, malformed = parse_source(source)
    cleaned, audit, quarantine, stats, group_label_support, low_support_labels = clean_records(records)
    write_cleaned_data(output, cleaned)

    stem = output.stem
    audit_path = output.with_name(f'{stem}_audit.csv')
    quarantine_path = output.with_name(f'{stem}_quarantine.csv')
    review_path = output.with_name(f'{stem}_label_review.csv')
    report_path = output.with_name(f'{stem}_report.json')
    audit_fields = [
        'action', 'reason', 'source_record_ids', 'source_line_numbers',
        'symptoms', 'original_label_variants', 'cleaned_labels',
    ]
    quarantine_fields = ['record_id', 'line_no', 'reason', 'symptom_count', 'symptoms', 'labels']
    write_csv(audit_path, audit, audit_fields)
    write_csv(quarantine_path, quarantine, quarantine_fields)
    label_review = build_label_review(records, cleaned, group_label_support, low_support_labels)
    write_csv(review_path, label_review, [
        'label', 'raw_support', 'unique_group_support', 'cleaned_support', 'status', 'note',
    ])

    removed_metadata = Counter(
        symptom for record in records for symptom in record['removed_symptoms']
    )
    report = {
        'cleaning_protocol_version': 'lhz_clean_v2',
        'source_path': str(source),
        'source_header': header,
        'source_sha256': file_sha256(source),
        'output_path': str(output),
        'output_sha256': file_sha256(output),
        'raw_valid_record_count': len(records),
        'malformed_line_count': len(malformed),
        **stats,
        'removed_non_symptom_metadata': dict(removed_metadata),
        'invalid_labels_removed': sorted(INVALID_LABELS),
        'label_aliases': LABEL_ALIASES,
        'rules': [
            'Unicode NFKC、去空白、单条记录内 token 去重并排序。',
            '删除数字时长和缺单位的介入术后数字字段，它们不是刻下症。',
            '单标签记录的症状数若大于清洗后全体记录P95，则直接删除，不猜测缺失标签。',
            '相同症状组的标签冲突取所有标注集合的交集，不使用并集。',
            '不同症状组合即使标签结果相同也分别保留，不做反向合并或约束。',
            '唯一症状组支持度小于5的标签直接删除；删除后无标签的记录整条删除。',
            '完全重复记录按用户要求保留；划分时仍必须以症状组为不可拆分单元。',
        ],
        'artifacts': {
            'audit_csv': str(audit_path),
            'removed_records_csv': str(quarantine_path),
            'label_review_csv': str(review_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'原始数据保持不变: {source}')
    print(f'清洗后数据: {output}')
    print(f'有效原始记录: {len(records)} -> 清洗记录: {len(cleaned)}')
    print(f'删除记录: {len(quarantine)}，删除低支持度标签: {len(low_support_labels)}')
    print(f'保留重复记录: {stats["exact_duplicate_excess_count_retained"]}，同症冲突组: {stats["conflicting_symptom_group_count"]}')
    print(f'审计文件: {audit_path}')
    print(f'清洗报告: {report_path}')
    print('注意：更换训练数据会改变实验协议，正式结论需要在清洗数据上重新运行主模型和消融。')
    return report


def build_parser():
    parser = argparse.ArgumentParser(description='生成 LHZ 可审计清洗数据集')
    parser.add_argument('--source', default=str(DEFAULT_SOURCE), help='原始三列 txt，绝不会覆盖')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='清洗后三列 txt')
    return parser


def main():
    args = build_parser().parse_args()
    run_cleaning(args.source, args.output)


if __name__ == '__main__':
    main()
