# -*- coding: utf-8 -*-
"""输出 18 个证候要素在清洗数据集中的分布表格，同时生成 LaTeX 代码。"""

import unicodedata
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[3] / 'dataset' / 'lhz_data_cleaned.txt'


def normalize_tokens(tokens):
    return tuple(sorted({
        unicodedata.normalize('NFKC', t).strip()
        for t in tokens if t.strip()
    }))


def load_cleaned(path):
    records = []
    with open(path, encoding='utf-8') as f:
        f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split(maxsplit=2)
            if len(parts) != 3:
                continue
            labels = normalize_tokens(parts[2].split(','))
            symptoms = normalize_tokens(parts[1].split(','))
            records.append({'symptoms': symptoms, 'labels': labels})
    return records


def build_table(records):
    n_records = len(records)
    n_groups = len({r['symptoms'] for r in records})

    label_record_count = Counter()   # 含该标签的记录数
    label_occurrence= Counter()   # 标签总出现次数（一条记录可含多个标签）
    label_count_per_record = Counter()  # 每条记录标签数（用于统计平均）

    for r in records:
        label_count_per_record[len(r['labels'])] += 1
        for lbl in r['labels']:
            label_record_count[lbl] += 1
            label_occurrence[lbl] += 1

    # 按记录数降序
    all_labels = sorted(label_record_count, key=lambda x: -label_record_count[x])

    print(f'清洗后总记录数 N = {n_records}，唯一症状组 = {n_groups}，标签种类 K = {len(all_labels)}')
    avg_labels = sum(k * v for k, v in label_count_per_record.items()) / n_records
    print(f'每条记录平均标签数 = {avg_labels:.2f}\n')

    # ── 控制台表格 ──
    header = f"{'证候要素':<10}{'记录数':>6}  {'占比%':>7}  {'总出现次':>8}"
    print(header)
    print('-' * len(header))
    for lbl in all_labels:
        cnt= label_record_count[lbl]
        occ  = label_occurrence[lbl]
        pct  = cnt / n_records * 100
        print(f'{lbl:<10}  {cnt:>6}  {pct:>7.2f}  {occ:>8}')

    # ── LaTeX 表格 ──
    print('\n\n%---- LaTeX ----')
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{清洗后数据集18个证候要素分布}')
    print(r'\label{tab:label_distribution}')
    print(r'\begin{tabular}{llrr}')
    print(r'\toprule')
    print(r'序号 & 证候要素 & 记录数 & 占比（\%）\\')
    print(r'\midrule')
    for i, lbl in enumerate(all_labels, 1):
        cnt = label_record_count[lbl]
        pct = cnt / n_records * 100
        print(f'{i} & {lbl} & {cnt} & {pct:.2f} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    records = load_cleaned(DATA_PATH)
    build_table(records)
