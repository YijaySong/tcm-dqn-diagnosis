# -*- coding: utf-8 -*-
"""纯监督基线：输入刻下症，输出证候要素Top-2。"""

import os
import math
import random
from collections import Counter
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import importlib
    sklearn_metrics = importlib.import_module('sklearn.metrics')
    sklearn_model_selection = importlib.import_module('sklearn.model_selection')
    precision_recall_fscore_support = sklearn_metrics.precision_recall_fscore_support
    hamming_loss = sklearn_metrics.hamming_loss
    train_test_split = sklearn_model_selection.train_test_split
except ImportError:
    precision_recall_fscore_support = None
    hamming_loss = None
    train_test_split = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_logger(logname):
    import logging
    logger = logging.getLogger(logname)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(logname, mode='w+')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s:  %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_path(filename):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', filename)


def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        file.readline()
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f'数据列数应为4列，实际为{len(parts)}列：{line}')
                data.append((parts[1].split(','), parts[2].split(',')))
    return data


def split_data(data, seed, test_ratio=0.1):
    combo_keys = ['|'.join(sorted(labels)) for _, labels in data]
    combo_counts = Counter(combo_keys)
    if train_test_split is not None and combo_counts and min(combo_counts.values()) >= 2:
        train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=seed, stratify=combo_keys)
        return list(train_data), list(test_data)

    rng = random.Random(seed)
    grouped = {}
    for item, key in zip(data, combo_keys):
        grouped.setdefault(key, []).append(item)
    train_data, test_data = [], []
    for items in grouped.values():
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_ratio))) if len(items) >= 2 else 0
        n_test = min(n_test, len(items) - 1) if len(items) >= 2 else 0
        test_data.extend(items[:n_test])
        train_data.extend(items[n_test:])
    rng.shuffle(train_data)
    rng.shuffle(test_data)
    return train_data, test_data


def build_vocab(train_data):
    symptom_to_idx = {}
    label_to_idx = {}
    for symptoms, labels in train_data:
        for symptom in symptoms:
            if symptom not in symptom_to_idx:
                symptom_to_idx[symptom] = len(symptom_to_idx)
        for label in labels:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
    return symptom_to_idx, label_to_idx


def vectorize(samples, symptom_to_idx, label_to_idx):
    x = np.zeros((len(samples), len(symptom_to_idx)), dtype=np.float32)
    y = np.zeros((len(samples), len(label_to_idx)), dtype=np.float32)
    for row_idx, (symptoms, labels) in enumerate(samples):
        for symptom in symptoms:
            if symptom in symptom_to_idx:
                x[row_idx, symptom_to_idx[symptom]] = 1.0
        for label in labels:
            if label in label_to_idx:
                y[row_idx, label_to_idx[label]] = 1.0
    return x, y


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def predict_topk(logits, top_k=2):
    top_k = min(top_k, logits.shape[1])
    return np.argsort(-logits, axis=1)[:, :top_k]


def predict_threshold(logits, threshold=0.5):
    probs = 1 / (1 + np.exp(-logits))
    preds = []
    for row in probs:
        idxs = np.where(row >= threshold)[0].tolist()
        if not idxs:
            idxs = [int(np.argmax(row))]
        preds.append(idxs)
    return preds


def to_multihot(predictions, num_labels):
    arr = np.zeros((len(predictions), num_labels), dtype=int)
    for row_idx, items in enumerate(predictions):
        for item in items:
            arr[row_idx, item] = 1
    return arr


def metrics(y_true, y_pred):
    sample_j, sample_f, exact = [], [], []
    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row > 0.5)[0].tolist())
        pred_set = set(np.where(pred_row > 0)[0].tolist())
        inter = len(true_set & pred_set)
        union = len(true_set | pred_set)
        p = inter / len(pred_set) if pred_set else 0.0
        r = inter / len(true_set) if true_set else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        sample_j.append(inter / union if union else 0.0)
        sample_f.append(f)
        exact.append(1 if pred_set == true_set else 0)
    if precision_recall_fscore_support is not None and hamming_loss is not None:
        micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        ham = hamming_loss(y_true, y_pred)
    else:
        tp = (y_true * y_pred).sum(axis=0)
        fp = ((1 - y_true) * y_pred).sum(axis=0)
        fn = (y_true * (1 - y_pred)).sum(axis=0)
        micro_p = tp.sum() / max(1, tp.sum() + fp.sum())
        micro_r = tp.sum() / max(1, tp.sum() + fn.sum())
        micro_f = 2 * micro_p * micro_r / max(1e-8, micro_p + micro_r)
        label_p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        label_r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        macro_p = float(np.mean(label_p))
        macro_r = float(np.mean(label_r))
        macro_f = float(np.mean(np.divide(2 * label_p * label_r, label_p + label_r, out=np.zeros_like(label_p, dtype=float), where=(label_p + label_r) != 0)))
        ham = float(np.not_equal(y_true, y_pred).mean())
    return {
        'sample_j': float(np.mean(sample_j)),
        'sample_f': float(np.mean(sample_f)),
        'exact': float(np.mean(exact)),
        'micro_f': float(micro_f),
        'macro_f': float(macro_f),
        'hamming': float(ham),
    }


def main():
    parser = ArgumentParser(description='Supervised baseline for syndrome element recommendation')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mode', choices=['top2', 'threshold'], default='top2')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = create_logger(f"supervised_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    data = load_data(build_path('广安门冠心病辨证数据集（不含主述）.txt'))
    train_data, test_data = split_data(data, args.seed, test_ratio=0.1)
    symptom_to_idx, label_to_idx = build_vocab(train_data)
    x_train, y_train = vectorize(train_data, symptom_to_idx, label_to_idx)
    x_test, y_test = vectorize(test_data, symptom_to_idx, label_to_idx)

    model = Classifier(len(symptom_to_idx), len(label_to_idx), args.hidden_dim, args.dropout)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(x_train_t))
        total_loss = 0.0
        for start in range(0, len(x_train_t), args.batch_size):
            idx = permutation[start:start + args.batch_size]
            batch_x = x_train_t[idx]
            batch_y = y_train_t[idx]
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        logger.info(f'epoch {epoch + 1}/{args.epochs}, loss={total_loss / len(x_train_t):.6f}')

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32)).cpu().numpy()

    if args.mode == 'top2':
        pred_idx = predict_topk(logits, top_k=2)
    else:
        pred_idx = [np.array(row) for row in predict_threshold(logits, threshold=args.threshold)]

    y_pred = to_multihot(pred_idx, len(label_to_idx))
    result = metrics(y_test, y_pred)

    logger.info(f"mode={args.mode}")
    logger.info(f"sample_j={result['sample_j']:.4f}, sample_f={result['sample_f']:.4f}, exact={result['exact']:.4f}")
    logger.info(f"micro_f={result['micro_f']:.4f}, macro_f={result['macro_f']:.4f}, hamming={result['hamming']:.4f}")
    print('done')


if __name__ == '__main__':
    main()
