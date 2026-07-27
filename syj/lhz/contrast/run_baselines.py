# -*- coding: utf-8 -*-
"""统一多标签基线实验入口。

建议保存位置：
    syj/lhz/contrast/run_baselines.py

运行示例：
    python3 syj/lhz/contrast/run_baselines.py --models all --seeds 9,17,29 --device auto
"""

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
LHZ_DIR = SCRIPT_DIR.parent
MYMODEL_DIR = LHZ_DIR / "mymodel_v1"
PROJECT_ROOT = SCRIPT_DIR.parents[2]

if str(MYMODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MYMODEL_DIR))

from constants import resolve_device
from data import (
    build_split_manifest,
    export_split_data,
    fit_training_schema,
    group_split_data,
    load_raw_tcm_data,
    transform_split,
)
from utils import configure_accelerator, create_logger, set_seed


MODEL_NAMES = (
    "frequency",
    "br_svm",
    "mlknn",
    "mlp_bce",
    "mlp_cb",
    "set_transformer",
    "stacv2",
    "ptm_top3",

)


MODEL_DISPLAY_NAMES = {
    "frequency": "Label Frequency",
    "br_svm": "BR + Linear SVM",
    "mlknn": "ML-kNN",
    "mlp_bce": "MLP + BCE",
    "mlp_cb": "MLP + CB-BCE",
    "set_transformer": "Set Transformer",
    "stacv2": "StaCv2 + LR",
    "ptm_top3": "PTM (Fixed Top-3)",

}


SUMMARY_METRICS = (
    "sample_jaccard",
    "sample_precision",
    "sample_recall",
    "sample_f1",
    "exact_match",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "supported_macro_precision",
    "supported_macro_recall",
    "supported_macro_f1",
    "zero_f1_count",
    "zero_f1_rate",
    "hit_rate",
    "empty_prediction_rate",
    "avg_selected_count",
    "avg_true_count",
    "avg_cardinality_error",
    "avg_over_select",
    "avg_under_select",
    "false_selection_rate",
    "miss_selection_rate",
    "premature_stop_rate",
    "late_stop_rate",
)

PRIMARY_METRICS = (
    "sample_f1",
    "micro_f1",
    "macro_f1",
    "supported_macro_f1",
    "exact_match",
    "sample_precision",
    "sample_recall",
    "miss_selection_rate",
    "zero_f1_count",
    "avg_selected_count",
)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)

    if fieldnames is None:
        # 不同模型的结果字段可能不同，例如：
        # best_k、best_c、best_threshold、elapsed_seconds。
        # 因此必须收集所有行的字段并集，不能只使用第一行。
        fieldnames = []
        seen_fields = set()

        for row in rows:
            for field_name in row.keys():
                if field_name not in seen_fields:
                    seen_fields.add(field_name)
                    fieldnames.append(field_name)

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            normalized_row = {
                field_name: json_ready(row.get(field_name, ""))
                for field_name in fieldnames
            }
            writer.writerow(normalized_row)



def set_all_seeds(seed):
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sigmoid_numpy(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def band_for_support(support):
    support = int(support)
    if support <= 4:
        return "rare_1_4"
    if support <= 19:
        return "few_5_19"
    if support <= 99:
        return "medium_20_99"
    return "head_100_plus"


def encode_dataset(records, symptom_to_index, label_to_index):
    row_indices = []
    column_indices = []
    values = []

    y = np.zeros((len(records), len(label_to_index)), dtype=np.int64)
    symptom_sequences = []
    sample_ids = []
    group_keys = []
    symptom_names = []
    truth_names = []

    for row_index, item in enumerate(records):
        symptoms, labels, record_id, group_key = item

        known_symptom_indices = sorted({
            symptom_to_index[symptom]
            for symptom in symptoms
            if symptom in symptom_to_index
        })

        symptom_sequences.append(known_symptom_indices)
        sample_ids.append(record_id)
        group_keys.append(group_key)
        symptom_names.append(list(symptoms))
        truth_names.append(list(labels))

        for symptom_index in known_symptom_indices:
            row_indices.append(row_index)
            column_indices.append(symptom_index)
            values.append(1.0)

        for label in labels:
            if label in label_to_index:
                y[row_index, label_to_index[label]] = 1

    x_sparse = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(records), len(symptom_to_index)),
        dtype=np.float32,
    )

    return {
        "x_sparse": x_sparse,
        "y": y,
        "sequences": symptom_sequences,
        "sample_ids": sample_ids,
        "group_keys": group_keys,
        "symptom_names": symptom_names,
        "truth_names": truth_names,
    }


def prepare_data(args, logger, output_dir):
    raw_records = load_raw_tcm_data(args.data_path, logger)

    raw_splits = group_split_data(
        raw_records,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        logger=logger,
    )

    schema = fit_training_schema(
        raw_splits["train"],
        max_se_num=args.max_se_num,
        logger=logger,
    )

    transformed = {}
    transform_stats = {}

    for split_name, records in raw_splits.items():
        transformed[split_name], transform_stats[split_name] = transform_split(
            records,
            schema,
            split_name,
            unknown_label_policy=args.unknown_label_policy,
        )

    manifest = build_split_manifest(
        args.data_path,
        args.split_seed,
        (args.train_ratio, args.val_ratio, args.test_ratio),
        raw_records,
        raw_splits,
        schema,
        transform_stats,
    )

    split_dir = output_dir / "shared_split"
    export_split_data(transformed, split_dir, manifest, logger)

    symptom_to_index = {
        symptom: int(index)
        for index, symptom in schema["symptoms"].items()
    }
    label_to_index = {
        label: int(index)
        for index, label in schema["labels"].items()
    }

    label_names = [
        schema["labels"][index]
        for index in range(len(schema["labels"]))
    ]

    encoded = {
        split_name: encode_dataset(
            records,
            symptom_to_index,
            label_to_index,
        )
        for split_name, records in transformed.items()
    }

    train_support = encoded["train"]["y"].sum(axis=0).astype(np.int64)

    return {
        "schema": schema,
        "manifest": manifest,
        "transformed": transformed,
        "encoded": encoded,
        "symptom_to_index": symptom_to_index,
        "label_to_index": label_to_index,
        "label_names": label_names,
        "train_support": train_support,
    }


def ensure_nonempty_predictions(binary_predictions, scores):
    binary_predictions = np.asarray(binary_predictions, dtype=np.int64).copy()
    scores = np.asarray(scores, dtype=np.float64)

    if binary_predictions.ndim != 2:
        raise ValueError("预测矩阵必须为二维")

    empty_rows = np.where(binary_predictions.sum(axis=1) == 0)[0]

    for row_index in empty_rows:
        best_label = int(np.argmax(scores[row_index]))
        binary_predictions[row_index, best_label] = 1

    return binary_predictions


def threshold_predictions(scores, threshold, min_labels=1):
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= float(threshold)).astype(np.int64)

    if int(min_labels) > 0:
        predictions = ensure_nonempty_predictions(predictions, scores)

    return predictions


def top_k_predictions(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    n_samples, n_labels = scores.shape

    k = max(1, min(int(k), n_labels))
    predictions = np.zeros((n_samples, n_labels), dtype=np.int64)

    top_indices = np.argpartition(
        -scores,
        kth=k - 1,
        axis=1,
    )[:, :k]

    row_indices = np.arange(n_samples)[:, None]
    predictions[row_indices, top_indices] = 1

    return predictions


def evaluate_binary_predictions(
    y_true,
    y_pred,
    label_names,
    train_support,
):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true.shape={y_true.shape} 与 y_pred.shape={y_pred.shape} 不一致"
        )

    n_samples, n_labels = y_true.shape

    sample_tp = (y_true * y_pred).sum(axis=1)
    sample_fp = ((1 - y_true) * y_pred).sum(axis=1)
    sample_fn = (y_true * (1 - y_pred)).sum(axis=1)

    sample_precision = safe_divide(sample_tp, sample_tp + sample_fp)
    sample_recall = safe_divide(sample_tp, sample_tp + sample_fn)
    sample_f1 = safe_divide(
        2.0 * sample_precision * sample_recall,
        sample_precision + sample_recall,
    )
    sample_jaccard = safe_divide(
        sample_tp,
        sample_tp + sample_fp + sample_fn,
    )

    exact_match = np.all(y_true == y_pred, axis=1).astype(np.float64)
    hit_rate = (sample_tp > 0).astype(np.float64)
    empty_prediction = (y_pred.sum(axis=1) == 0).astype(np.float64)

    selected_counts = y_pred.sum(axis=1)
    true_counts = y_true.sum(axis=1)

    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    label_precision = safe_divide(tp, tp + fp)
    label_recall = safe_divide(tp, tp + fn)
    label_f1 = safe_divide(
        2.0 * label_precision * label_recall,
        label_precision + label_recall,
    )

    eval_support = y_true.sum(axis=0)
    pred_count = y_pred.sum(axis=0)
    supported = eval_support > 0

    total_tp = float(tp.sum())
    total_fp = float(fp.sum())
    total_fn = float(fn.sum())

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )
    micro_f1 = (
        2.0 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    label_metrics = []

    for label_index in range(n_labels):
        support = int(train_support[label_index])

        label_metrics.append({
            "label": label_names[label_index],
            "train_support": support,
            "eval_support": int(eval_support[label_index]),
            "pred_count": int(pred_count[label_index]),
            "tp": int(tp[label_index]),
            "fp": int(fp[label_index]),
            "fn": int(fn[label_index]),
            "precision": float(label_precision[label_index]),
            "recall": float(label_recall[label_index]),
            "f1": float(label_f1[label_index]),
            "frequency_band": band_for_support(support),
        })

    band_metrics = {}

    for band_name in (
        "rare_1_4",
        "few_5_19",
        "medium_20_99",
        "head_100_plus",
    ):
        rows = [
            row for row in label_metrics
            if row["frequency_band"] == band_name
        ]
        supported_rows = [
            row for row in rows
            if row["eval_support"] > 0
        ]

        band_metrics[band_name] = {
            "label_count": len(rows),
            "eval_supported_label_count": len(supported_rows),
            "macro_f1": (
                float(np.mean([row["f1"] for row in supported_rows]))
                if supported_rows
                else 0.0
            ),
            "macro_recall": (
                float(np.mean([row["recall"] for row in supported_rows]))
                if supported_rows
                else 0.0
            ),
            "zero_f1_count": int(sum(
                row["f1"] == 0.0 for row in supported_rows
            )),
            "never_predicted_count": int(sum(
                row["pred_count"] == 0 for row in supported_rows
            )),
        }

    supported_label_f1 = label_f1[supported]
    supported_label_precision = label_precision[supported]
    supported_label_recall = label_recall[supported]

    metrics = {
        "sample_jaccard": float(np.mean(sample_jaccard)),
        "sample_precision": float(np.mean(sample_precision)),
        "sample_recall": float(np.mean(sample_recall)),
        "sample_f1": float(np.mean(sample_f1)),
        "exact_match": float(np.mean(exact_match)),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean(label_precision)),
        "macro_recall": float(np.mean(label_recall)),
        "macro_f1": float(np.mean(label_f1)),
        "supported_macro_precision": (
            float(np.mean(supported_label_precision))
            if supported.any()
            else 0.0
        ),
        "supported_macro_recall": (
            float(np.mean(supported_label_recall))
            if supported.any()
            else 0.0
        ),
        "supported_macro_f1": (
            float(np.mean(supported_label_f1))
            if supported.any()
            else 0.0
        ),
        "supported_label_count": int(supported.sum()),
        "zero_f1_count": int(np.sum(supported_label_f1 == 0.0)),
        "zero_f1_rate": (
            float(np.mean(supported_label_f1 == 0.0))
            if supported.any()
            else 0.0
        ),
        "hit_rate": float(np.mean(hit_rate)),
        "empty_prediction_rate": float(np.mean(empty_prediction)),
        "avg_selected_count": float(np.mean(selected_counts)),
        "avg_true_count": float(np.mean(true_counts)),
        "avg_cardinality_error": float(np.mean(
            np.abs(selected_counts - true_counts)
        )),
        "avg_over_select": float(np.mean(
            np.maximum(0, selected_counts - true_counts)
        )),
        "avg_under_select": float(np.mean(
            np.maximum(0, true_counts - selected_counts)
        )),
        "false_selection_rate": float(np.mean(1.0 - sample_precision)),
        "miss_selection_rate": float(np.mean(1.0 - sample_recall)),
        "avg_stop_depth": float(np.mean(selected_counts)),
        "avg_stop_margin": 0.0,
        "premature_stop_rate": float(np.mean(
            selected_counts < true_counts
        )),
        "late_stop_rate": float(np.mean(
            selected_counts > true_counts
        )),
        "n_samples": int(n_samples),
    }

    return {
        "auto": metrics,
        "label_metrics": label_metrics,
        "band_metrics": band_metrics,
    }


def selection_key(metrics, threshold=None):
    auto = metrics["auto"]

    key = (
        float(auto["supported_macro_f1"]),
        float(auto["sample_f1"]),
        float(auto["exact_match"]),
        float(auto["micro_f1"]),
    )

    if threshold is not None:
        key = key + (-abs(float(threshold) - 0.5),)

    return key


def choose_best_threshold(
    scores,
    y_true,
    label_names,
    train_support,
    thresholds,
    min_labels=1,
):
    best = None
    history = []

    for threshold in thresholds:
        predictions = threshold_predictions(
            scores,
            threshold,
            min_labels=min_labels,
        )

        metrics = evaluate_binary_predictions(
            y_true,
            predictions,
            label_names,
            train_support,
        )

        row = {
            "threshold": float(threshold),
            **metrics["auto"],
        }
        history.append(row)

        candidate = {
            "threshold": float(threshold),
            "predictions": predictions,
            "metrics": metrics,
            "key": selection_key(metrics, threshold),
        }

        if best is None or candidate["key"] > best["key"]:
            best = candidate

    return best, history


def choose_best_k(
    scores,
    y_true,
    label_names,
    train_support,
    k_values,
):
    best = None
    history = []

    for k in k_values:
        predictions = top_k_predictions(scores, k)

        metrics = evaluate_binary_predictions(
            y_true,
            predictions,
            label_names,
            train_support,
        )

        row = {
            "k": int(k),
            **metrics["auto"],
        }
        history.append(row)

        candidate = {
            "k": int(k),
            "predictions": predictions,
            "metrics": metrics,
            "key": selection_key(metrics),
        }

        if best is None or candidate["key"] > best["key"]:
            best = candidate

    return best, history


def build_prediction_rows(
    split_data,
    y_true,
    y_pred,
    scores,
    label_names,
):
    rows = []

    for row_index in range(len(y_true)):
        true_indices = np.flatnonzero(y_true[row_index])
        pred_indices = np.flatnonzero(y_pred[row_index])

        rows.append({
            "record_id": split_data["sample_ids"][row_index],
            "symptom_group_key": split_data["group_keys"][row_index],
            "symptoms": ",".join(split_data["symptom_names"][row_index]),
            "true_labels": ",".join(
                label_names[index] for index in true_indices
            ),
            "predicted_labels": ",".join(
                label_names[index] for index in pred_indices
            ),
            "true_count": int(len(true_indices)),
            "predicted_count": int(len(pred_indices)),
            "predicted_scores": json.dumps(
                {
                    label_names[index]: round(float(scores[row_index, index]), 8)
                    for index in pred_indices
                },
                ensure_ascii=False,
            ),
        })

    return rows


def save_evaluation_artifacts(
    output_dir,
    split_name,
    split_data,
    scores,
    predictions,
    result,
    label_names,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / f"{split_name}_metrics.json",
        result,
    )

    write_csv(
        output_dir / f"{split_name}_label_metrics.csv",
        result["label_metrics"],
    )

    band_rows = [
        {
            "frequency_band": band_name,
            **band_values,
        }
        for band_name, band_values in result["band_metrics"].items()
    ]

    write_csv(
        output_dir / f"{split_name}_band_metrics.csv",
        band_rows,
    )

    prediction_rows = build_prediction_rows(
        split_data,
        split_data["y"],
        predictions,
        scores,
        label_names,
    )

    write_csv(
        output_dir / f"{split_name}_predictions.csv",
        prediction_rows,
    )


class FrequencyBaseline:
    def __init__(self):
        self.prevalence = None

    def fit(self, y_train):
        self.prevalence = np.asarray(
            y_train,
            dtype=np.float64,
        ).mean(axis=0)
        return self

    def predict_scores(self, n_samples):
        return np.repeat(
            self.prevalence.reshape(1, -1),
            repeats=int(n_samples),
            axis=0,
        )


class BinaryRelevanceLinearSVM:
    def __init__(self, c=1.0, seed=9):
        self.c = float(c)
        self.seed = int(seed)
        self.models = []
        self.constant_scores = []

    def fit(self, x_train, y_train):
        self.models = []
        self.constant_scores = []

        for label_index in range(y_train.shape[1]):
            targets = y_train[:, label_index]
            unique_values = np.unique(targets)

            if len(unique_values) == 1:
                self.models.append(None)
                self.constant_scores.append(
                    20.0 if int(unique_values[0]) == 1 else -20.0
                )
                continue

            model = LinearSVC(
                C=self.c,
                class_weight=None,
                random_state=self.seed,
                max_iter=20000,
                dual=True,
            )
            model.fit(x_train, targets)

            self.models.append(model)
            self.constant_scores.append(None)

        return self

    def decision_function(self, x):
        score_columns = []

        for model, constant_score in zip(
            self.models,
            self.constant_scores,
        ):
            if model is None:
                scores = np.full(
                    x.shape[0],
                    float(constant_score),
                    dtype=np.float64,
                )
            else:
                scores = np.asarray(
                    model.decision_function(x),
                    dtype=np.float64,
                ).reshape(-1)

            score_columns.append(scores)

        return np.stack(score_columns, axis=1)

    def predict_scores(self, x):
        return sigmoid_numpy(self.decision_function(x))


class MLKNN:
    def __init__(
        self,
        n_neighbors=10,
        smoothing=1.0,
        metric="cosine",
        n_jobs=-1,
    ):
        self.n_neighbors = int(n_neighbors)
        self.smoothing = float(smoothing)
        self.metric = str(metric)
        self.n_jobs = int(n_jobs)

        self.x_train = None
        self.y_train = None
        self.neighbor_model = None
        self.prior_positive = None
        self.prior_negative = None
        self.conditional_positive = None
        self.conditional_negative = None
        self.effective_k = None

    def _training_neighbor_indices(self):
        n_samples = self.x_train.shape[0]
        query_neighbors = min(self.effective_k + 1, n_samples)

        _, raw_indices = self.neighbor_model.kneighbors(
            self.x_train,
            n_neighbors=query_neighbors,
            return_distance=True,
        )

        output = np.zeros(
            (n_samples, self.effective_k),
            dtype=np.int64,
        )

        for row_index, indices in enumerate(raw_indices):
            filtered = [
                int(index)
                for index in indices
                if int(index) != row_index
            ]

            if len(filtered) < self.effective_k:
                filtered.extend([
                    int(indices[-1])
                ] * (self.effective_k - len(filtered)))

            output[row_index] = np.asarray(
                filtered[:self.effective_k],
                dtype=np.int64,
            )

        return output

    def fit(self, x_train, y_train):
        self.x_train = sparse.csr_matrix(
            x_train,
            dtype=np.float32,
        )
        self.y_train = np.asarray(
            y_train,
            dtype=np.int64,
        )

        n_samples = self.x_train.shape[0]

        if n_samples < 2:
            raise ValueError("ML-kNN 至少需要两个训练样本")

        self.effective_k = max(
            1,
            min(self.n_neighbors, n_samples - 1),
        )

        self.neighbor_model = NearestNeighbors(
            n_neighbors=self.effective_k + 1,
            metric=self.metric,
            algorithm="brute",
            n_jobs=self.n_jobs,
        )
        self.neighbor_model.fit(self.x_train)

        train_neighbor_indices = self._training_neighbor_indices()
        neighbor_counts = self.y_train[
            train_neighbor_indices
        ].sum(axis=1)

        n_labels = self.y_train.shape[1]
        smoothing = self.smoothing

        positive_counts = self.y_train.sum(axis=0).astype(np.float64)
        negative_counts = n_samples - positive_counts

        self.prior_positive = (
            smoothing + positive_counts
        ) / (
            2.0 * smoothing + n_samples
        )
        self.prior_negative = 1.0 - self.prior_positive

        self.conditional_positive = np.zeros(
            (n_labels, self.effective_k + 1),
            dtype=np.float64,
        )
        self.conditional_negative = np.zeros(
            (n_labels, self.effective_k + 1),
            dtype=np.float64,
        )

        for label_index in range(n_labels):
            positive_mask = self.y_train[:, label_index] == 1
            negative_mask = ~positive_mask

            positive_histogram = np.bincount(
                neighbor_counts[positive_mask, label_index],
                minlength=self.effective_k + 1,
            ).astype(np.float64)

            negative_histogram = np.bincount(
                neighbor_counts[negative_mask, label_index],
                minlength=self.effective_k + 1,
            ).astype(np.float64)

            self.conditional_positive[label_index] = (
                smoothing + positive_histogram
            ) / (
                smoothing * (self.effective_k + 1)
                + positive_counts[label_index]
            )

            self.conditional_negative[label_index] = (
                smoothing + negative_histogram
            ) / (
                smoothing * (self.effective_k + 1)
                + negative_counts[label_index]
            )

        return self

    def predict_scores(self, x):
        x = sparse.csr_matrix(x, dtype=np.float32)

        _, neighbor_indices = self.neighbor_model.kneighbors(
            x,
            n_neighbors=self.effective_k,
            return_distance=True,
        )

        neighbor_counts = self.y_train[
            neighbor_indices
        ].sum(axis=1)

        n_samples = x.shape[0]
        n_labels = self.y_train.shape[1]
        probabilities = np.zeros(
            (n_samples, n_labels),
            dtype=np.float64,
        )

        for label_index in range(n_labels):
            count_values = neighbor_counts[:, label_index].astype(np.int64)

            positive_joint = (
                self.prior_positive[label_index]
                * self.conditional_positive[
                    label_index,
                    count_values,
                ]
            )

            negative_joint = (
                self.prior_negative[label_index]
                * self.conditional_negative[
                    label_index,
                    count_values,
                ]
            )

            denominator = positive_joint + negative_joint

            probabilities[:, label_index] = np.divide(
                positive_joint,
                denominator,
                out=np.full_like(
                    positive_joint,
                    self.prior_positive[label_index],
                ),
                where=denominator > 0,
            )

        return probabilities


class ConstantBinaryProbabilityModel:
    """处理训练子集中只有一个类别的二分类标签。

    长尾证素在某些交叉验证训练折中可能全部为0或全部为1，
    此时LogisticRegression无法正常训练，因此使用常量模型。
    """

    def __init__(self, value):
        self.value = int(value)

    def predict_scores(self, x):
        return np.full(
            x.shape[0],
            float(self.value),
            dtype=np.float64,
        )


class StaCv2:
    """StaCv2多标签分类器。

    第一层
    ------
    使用Classifier Chain。训练当前标签分类器时，将前序标签的
    真实值加入特征；推理时，则将前序标签的预测值加入特征。

    第二层
    ------
    首先使用第一层的所有标签预测构建标签状态。预测当前标签时，
    使用输入特征和除当前标签外的标签状态作为输入。当前标签完成
    第二层预测后，用第二层预测替换对应的第一层预测，供后续标签
    分类器使用。

    为避免直接使用训练集内预测导致的数据泄漏，第二层训练所需的
    第一层预测和前序第二层预测均通过K折交叉验证生成。

    Parameters
    ----------
    c:
        Logistic Regression的正则化参数。

    n_folds:
        生成OOF预测时使用的交叉验证折数。

    internal_threshold:
        链式预测过程中将概率转换成标签状态的内部阈值。
        最终输出概率仍会通过验证集重新搜索全局预测阈值。

    max_iter:
        Logistic Regression的最大迭代次数。

    seed:
        随机种子，同时控制标签顺序和交叉验证划分。
    """

    def __init__(
        self,
        c=1.0,
        n_folds=3,
        internal_threshold=0.5,
        max_iter=1000,
        seed=9,
    ):
        self.c = float(c)
        self.n_folds = int(n_folds)
        self.internal_threshold = float(internal_threshold)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

        self.n_features = None
        self.n_labels = None
        self.label_order = None

        self.first_layer_models = None
        self.second_layer_models = None

    @staticmethod
    def _to_dense(x):
        """将输入统一转换为二维float64稠密矩阵。"""
        if sparse.issparse(x):
            x = x.toarray()

        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError(
                f"StaCv2输入必须是二维矩阵，当前shape={x.shape}"
            )

        return x

    def _build_logistic_regression(self, seed_offset=0):
        return LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=self.c,
            tol=1e-4,
            max_iter=self.max_iter,
            random_state=self.seed + int(seed_offset),
        )

    def _fit_binary_model(
        self,
        x,
        targets,
        seed_offset=0,
    ):
        targets = np.asarray(
            targets,
            dtype=np.int64,
        ).reshape(-1)

        unique_values = np.unique(targets)

        if len(unique_values) == 1:
            return ConstantBinaryProbabilityModel(
                unique_values[0]
            )

        model = self._build_logistic_regression(
            seed_offset=seed_offset,
        )
        model.fit(x, targets)
        return model

    @staticmethod
    def _predict_binary_scores(model, x):
        if isinstance(
            model,
            ConstantBinaryProbabilityModel,
        ):
            return model.predict_scores(x)

        probabilities = np.asarray(
            model.predict_proba(x),
            dtype=np.float64,
        )

        # 正常二分类LogisticRegression返回[n_samples, 2]。
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]

        return probabilities.reshape(-1)

    def _scores_to_binary(self, scores):
        return (
            np.asarray(scores, dtype=np.float64)
            >= self.internal_threshold
        ).astype(np.int64)

    def _create_kfold(self, n_samples, seed_offset=0):
        n_splits = min(
            int(self.n_folds),
            int(n_samples),
        )

        if n_splits < 2:
            raise ValueError(
                "StaCv2的OOF预测至少需要两个训练样本"
            )

        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.seed + int(seed_offset),
        )

    def _fit_first_layer(self, x, y):
        """训练完整的第一层Classifier Chain。

        训练过程中使用前序标签真值扩展特征。
        """
        models = [None] * self.n_labels
        current_features = np.asarray(
            x,
            dtype=np.float64,
        )

        for position, label_index in enumerate(
            self.label_order
        ):
            targets = y[:, label_index]

            model = self._fit_binary_model(
                current_features,
                targets,
                seed_offset=100 + position,
            )
            models[label_index] = model

            # 训练阶段使用真实标签。
            current_features = np.column_stack([
                current_features,
                targets,
            ])

        return models

    def _predict_first_layer(
        self,
        x,
        models,
    ):
        """使用第一层Classifier Chain进行预测。

        推理过程中使用前序标签的预测结果扩展特征。
        """
        n_samples = x.shape[0]

        scores = np.zeros(
            (n_samples, self.n_labels),
            dtype=np.float64,
        )
        predictions = np.zeros(
            (n_samples, self.n_labels),
            dtype=np.int64,
        )

        current_features = np.asarray(
            x,
            dtype=np.float64,
        )

        for label_index in self.label_order:
            model = models[label_index]

            label_scores = self._predict_binary_scores(
                model,
                current_features,
            )
            label_predictions = self._scores_to_binary(
                label_scores
            )

            scores[:, label_index] = label_scores
            predictions[:, label_index] = label_predictions

            # 推理阶段使用预测标签。
            current_features = np.column_stack([
                current_features,
                label_predictions,
            ])

        return scores, predictions

    def _generate_first_layer_oof(self, x, y):
        """生成第一层Classifier Chain的OOF预测。"""
        n_samples = x.shape[0]

        oof_scores = np.zeros(
            (n_samples, self.n_labels),
            dtype=np.float64,
        )
        oof_predictions = np.zeros(
            (n_samples, self.n_labels),
            dtype=np.int64,
        )

        kfold = self._create_kfold(
            n_samples,
            seed_offset=1000,
        )

        for fold_index, (
            train_indices,
            validation_indices,
        ) in enumerate(kfold.split(x)):
            fold_models = self._fit_first_layer(
                x[train_indices],
                y[train_indices],
            )

            fold_scores, fold_predictions = (
                self._predict_first_layer(
                    x[validation_indices],
                    fold_models,
                )
            )

            oof_scores[validation_indices] = fold_scores
            oof_predictions[validation_indices] = (
                fold_predictions
            )

        return oof_scores, oof_predictions

    @staticmethod
    def _build_second_layer_features(
        x,
        label_state,
        current_label,
    ):
        """构建第二层当前标签分类器的输入。

        当前标签自身的第一层预测会被移除，输入由以下两部分组成：

            [原始症状特征, 其余K-1个标签状态]
        """
        other_label_states = np.delete(
            label_state,
            int(current_label),
            axis=1,
        )

        return np.column_stack([
            x,
            other_label_states,
        ])

    def _generate_binary_oof(
        self,
        x,
        targets,
        seed_offset=0,
    ):
        """为第二层单个二分类器生成OOF预测。"""
        n_samples = x.shape[0]

        oof_scores = np.zeros(
            n_samples,
            dtype=np.float64,
        )
        oof_predictions = np.zeros(
            n_samples,
            dtype=np.int64,
        )

        kfold = self._create_kfold(
            n_samples,
            seed_offset=seed_offset,
        )

        for fold_index, (
            train_indices,
            validation_indices,
        ) in enumerate(kfold.split(x)):
            model = self._fit_binary_model(
                x[train_indices],
                targets[train_indices],
                seed_offset=(
                    seed_offset
                    + fold_index
                    + 1
                ),
            )

            fold_scores = self._predict_binary_scores(
                model,
                x[validation_indices],
            )
            fold_predictions = self._scores_to_binary(
                fold_scores
            )

            oof_scores[validation_indices] = fold_scores
            oof_predictions[validation_indices] = (
                fold_predictions
            )

        return oof_scores, oof_predictions

    def fit(self, x_train, y_train):
        x_train = self._to_dense(x_train)
        y_train = np.asarray(
            y_train,
            dtype=np.int64,
        )

        if y_train.ndim != 2:
            raise ValueError(
                f"StaCv2标签必须是二维矩阵，当前shape={y_train.shape}"
            )

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "StaCv2的输入样本数与标签样本数不一致"
            )

        if not np.all(np.isin(y_train, [0, 1])):
            raise ValueError(
                "StaCv2要求标签矩阵只能包含0和1"
            )

        self.n_features = int(x_train.shape[1])
        self.n_labels = int(y_train.shape[1])

        # StaCv2使用随机标签顺序。每个实验种子产生一个确定但不同
        # 的随机排列。
        random_generator = np.random.default_rng(
            self.seed
        )
        self.label_order = (
            random_generator.permutation(
                self.n_labels
            ).astype(np.int64).tolist()
        )

        # 第一层OOF预测用于第二层训练。
        _, first_layer_oof_predictions = (
            self._generate_first_layer_oof(
                x_train,
                y_train,
            )
        )

        # 在完整训练集上拟合最终第一层模型，用于验证集和测试集。
        self.first_layer_models = self._fit_first_layer(
            x_train,
            y_train,
        )

        # 第二层的初始标签状态为第一层OOF预测。
        second_layer_state = (
            first_layer_oof_predictions.copy()
        )

        self.second_layer_models = [
            None
        ] * self.n_labels

        for position, label_index in enumerate(
            self.label_order
        ):
            current_features = (
                self._build_second_layer_features(
                    x_train,
                    second_layer_state,
                    label_index,
                )
            )
            targets = y_train[:, label_index]

            # 为当前第二层分类器生成OOF预测，供后续第二层分类器
            # 使用，避免训练集内预测泄漏。
            _, current_oof_predictions = (
                self._generate_binary_oof(
                    current_features,
                    targets,
                    seed_offset=2000 + position * 100,
                )
            )

            # 在全部训练样本上拟合最终的当前第二层分类器。
            final_model = self._fit_binary_model(
                current_features,
                targets,
                seed_offset=3000 + position,
            )

            self.second_layer_models[label_index] = (
                final_model
            )

            # 用第二层OOF预测替换当前标签的第一层预测。
            second_layer_state[:, label_index] = (
                current_oof_predictions
            )

        return self

    def predict_scores(self, x):
        if self.first_layer_models is None:
            raise RuntimeError(
                "StaCv2尚未完成训练"
            )

        x = self._to_dense(x)

        if x.shape[1] != self.n_features:
            raise ValueError(
                f"StaCv2期望输入维度为{self.n_features}，"
                f"实际为{x.shape[1]}"
            )

        _, first_layer_predictions = (
            self._predict_first_layer(
                x,
                self.first_layer_models,
            )
        )

        # 第二层标签状态初始为第一层预测。
        second_layer_state = (
            first_layer_predictions.copy()
        )

        second_layer_scores = np.zeros(
            (x.shape[0], self.n_labels),
            dtype=np.float64,
        )

        for label_index in self.label_order:
            current_features = (
                self._build_second_layer_features(
                    x,
                    second_layer_state,
                    label_index,
                )
            )

            model = self.second_layer_models[
                label_index
            ]

            current_scores = (
                self._predict_binary_scores(
                    model,
                    current_features,
                )
            )
            current_predictions = (
                self._scores_to_binary(
                    current_scores
                )
            )

            second_layer_scores[
                :, label_index
            ] = current_scores

            # 用第二层预测替换对应的第一层预测。
            second_layer_state[
                :, label_index
            ] = current_predictions

        return second_layer_scores

class PTMFixedTopK:
    """面向症状—证素预测的PTM适配实现。

    症状对应原文中的症状词，证素对应待推荐项目。
    每个样本共享主题分布，证素侧保留4个潜在角色。
    """

    def __init__(
        self,
        n_topics=25,
        n_roles=4,
        alpha=1.0,
        beta_symptom=0.1,
        beta_label=0.1,
        eta=1.0,
        iterations=500,
        inference_iterations=100,
        seed=9,
    ):
        self.n_topics = int(n_topics)
        self.n_roles = int(n_roles)
        self.alpha = float(alpha)
        self.beta_symptom = float(beta_symptom)
        self.beta_label = float(beta_label)
        self.eta = float(eta)
        self.iterations = int(iterations)
        self.inference_iterations = int(inference_iterations)
        self.seed = int(seed)

        self.symptom_count = None
        self.label_count = None

        self.topic_symptom = None
        self.topic_symptom_total = None
        self.topic_role_label = None
        self.topic_role_total = None
        self.label_given_topic = None

    @staticmethod
    def _sample_index(probabilities, rng):
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )
        probabilities = np.maximum(probabilities, 0.0)

        total = float(probabilities.sum())

        if not np.isfinite(total) or total <= 0.0:
            return int(rng.integers(len(probabilities)))

        probabilities /= total
        return int(rng.choice(
            len(probabilities),
            p=probabilities,
        ))

    def fit(
        self,
        symptom_sequences,
        y_train,
        symptom_count,
    ):
        y_train = np.asarray(y_train, dtype=np.int64)

        if y_train.ndim != 2:
            raise ValueError("PTM要求二维标签矩阵")

        n_documents = len(symptom_sequences)

        if n_documents != y_train.shape[0]:
            raise ValueError("PTM症状序列与标签样本数不一致")

        self.symptom_count = int(symptom_count)
        self.label_count = int(y_train.shape[1])

        k_count = self.n_topics
        role_count = self.n_roles
        rng = np.random.default_rng(self.seed)

        document_topic = np.zeros(
            (n_documents, k_count),
            dtype=np.int64,
        )
        document_topic_role = np.zeros(
            (n_documents, k_count, role_count),
            dtype=np.int64,
        )

        self.topic_symptom = np.zeros(
            (k_count, self.symptom_count),
            dtype=np.int64,
        )
        self.topic_symptom_total = np.zeros(
            k_count,
            dtype=np.int64,
        )

        self.topic_role_label = np.zeros(
            (k_count, role_count, self.label_count),
            dtype=np.int64,
        )
        self.topic_role_total = np.zeros(
            (k_count, role_count),
            dtype=np.int64,
        )

        symptom_assignments = []
        label_assignments = []
        document_labels = []

        # 随机初始化症状主题和证素主题—角色。
        for document_index, sequence in enumerate(
            symptom_sequences
        ):
            sequence = np.asarray(
                sequence,
                dtype=np.int64,
            )

            symptom_topics = rng.integers(
                0,
                k_count,
                size=len(sequence),
                dtype=np.int64,
            )

            symptom_assignments.append(symptom_topics)

            for symptom_index, topic in zip(
                sequence,
                symptom_topics,
            ):
                document_topic[
                    document_index,
                    topic,
                ] += 1
                self.topic_symptom[
                    topic,
                    symptom_index,
                ] += 1
                self.topic_symptom_total[topic] += 1

            labels = np.flatnonzero(
                y_train[document_index]
            ).astype(np.int64)
            document_labels.append(labels)

            assignments = np.zeros(
                (len(labels), 2),
                dtype=np.int64,
            )

            for item_index, label_index in enumerate(labels):
                topic = int(rng.integers(k_count))
                role = int(rng.integers(role_count))

                assignments[item_index] = [topic, role]

                document_topic[
                    document_index,
                    topic,
                ] += 1
                document_topic_role[
                    document_index,
                    topic,
                    role,
                ] += 1
                self.topic_role_label[
                    topic,
                    role,
                    label_index,
                ] += 1
                self.topic_role_total[
                    topic,
                    role,
                ] += 1

            label_assignments.append(assignments)

        for iteration in range(self.iterations):
            for document_index, sequence in enumerate(
                symptom_sequences
            ):
                sequence = np.asarray(
                    sequence,
                    dtype=np.int64,
                )
                assignments = symptom_assignments[
                    document_index
                ]

                # 更新症状的主题分配。
                for token_index, symptom_index in enumerate(
                    sequence
                ):
                    old_topic = int(assignments[token_index])

                    document_topic[
                        document_index,
                        old_topic,
                    ] -= 1
                    self.topic_symptom[
                        old_topic,
                        symptom_index,
                    ] -= 1
                    self.topic_symptom_total[
                        old_topic
                    ] -= 1

                    topic_probability = (
                        document_topic[
                            document_index
                        ] + self.alpha
                    ) * (
                        self.topic_symptom[
                            :,
                            symptom_index,
                        ] + self.beta_symptom
                    ) / (
                        self.topic_symptom_total
                        + self.symptom_count
                        * self.beta_symptom
                    )

                    new_topic = self._sample_index(
                        topic_probability,
                        rng,
                    )

                    assignments[token_index] = new_topic

                    document_topic[
                        document_index,
                        new_topic,
                    ] += 1
                    self.topic_symptom[
                        new_topic,
                        symptom_index,
                    ] += 1
                    self.topic_symptom_total[
                        new_topic
                    ] += 1

                # 更新证素的主题和角色分配。
                labels = document_labels[document_index]
                assignments = label_assignments[
                    document_index
                ]

                for item_index, label_index in enumerate(labels):
                    old_topic = int(assignments[item_index, 0])
                    old_role = int(assignments[item_index, 1])

                    document_topic[
                        document_index,
                        old_topic,
                    ] -= 1
                    document_topic_role[
                        document_index,
                        old_topic,
                        old_role,
                    ] -= 1
                    self.topic_role_label[
                        old_topic,
                        old_role,
                        label_index,
                    ] -= 1
                    self.topic_role_total[
                        old_topic,
                        old_role,
                    ] -= 1

                    topic_factor = (
                        document_topic[
                            document_index
                        ] + self.alpha
                    )[:, None]

                    role_denominator = (
                        document_topic_role[
                            document_index
                        ].sum(axis=1)
                        + role_count * self.eta
                    )[:, None]

                    role_factor = (
                        document_topic_role[
                            document_index
                        ] + self.eta
                    ) / role_denominator

                    label_factor = (
                        self.topic_role_label[
                            :,
                            :,
                            label_index,
                        ] + self.beta_label
                    ) / (
                        self.topic_role_total
                        + self.label_count
                        * self.beta_label
                    )

                    joint_probability = (
                        topic_factor
                        * role_factor
                        * label_factor
                    ).reshape(-1)

                    sampled_index = self._sample_index(
                        joint_probability,
                        rng,
                    )

                    new_topic = sampled_index // role_count
                    new_role = sampled_index % role_count

                    assignments[item_index] = [
                        new_topic,
                        new_role,
                    ]

                    document_topic[
                        document_index,
                        new_topic,
                    ] += 1
                    document_topic_role[
                        document_index,
                        new_topic,
                        new_role,
                    ] += 1
                    self.topic_role_label[
                        new_topic,
                        new_role,
                        label_index,
                    ] += 1
                    self.topic_role_total[
                        new_topic,
                        new_role,
                    ] += 1

        # 计算主题下的角色概率。
        global_role_counts = self.topic_role_label.sum(
            axis=2
        ).astype(np.float64)

        role_given_topic = (
            global_role_counts + self.eta
        ) / (
            global_role_counts.sum(
                axis=1,
                keepdims=True,
            )
            + role_count * self.eta
        )

        label_given_topic_role = (
            self.topic_role_label.astype(np.float64)
            + self.beta_label
        ) / (
            self.topic_role_total[
                :,
                :,
                None,
            ]
            + self.label_count * self.beta_label
        )

        self.label_given_topic = np.einsum(
            "kr,krl->kl",
            role_given_topic,
            label_given_topic_role,
        )

        return self

    def predict_scores(
        self,
        symptom_sequences,
        seed_offset=0,
    ):
        if self.label_given_topic is None:
            raise RuntimeError("PTM尚未训练")

        rng = np.random.default_rng(
            self.seed + 10000 + int(seed_offset)
        )

        symptom_given_topic = (
            self.topic_symptom.astype(np.float64)
            + self.beta_symptom
        ) / (
            self.topic_symptom_total[:, None]
            + self.symptom_count * self.beta_symptom
        )

        output_scores = np.zeros(
            (len(symptom_sequences), self.label_count),
            dtype=np.float64,
        )

        for document_index, sequence in enumerate(
            symptom_sequences
        ):
            sequence = np.asarray(
                sequence,
                dtype=np.int64,
            )

            document_topic = np.zeros(
                self.n_topics,
                dtype=np.int64,
            )

            assignments = rng.integers(
                0,
                self.n_topics,
                size=len(sequence),
                dtype=np.int64,
            )

            for topic in assignments:
                document_topic[topic] += 1

            for _ in range(self.inference_iterations):
                for token_index, symptom_index in enumerate(
                    sequence
                ):
                    old_topic = int(assignments[token_index])
                    document_topic[old_topic] -= 1

                    probabilities = (
                        document_topic + self.alpha
                    ) * symptom_given_topic[
                        :,
                        symptom_index,
                    ]

                    new_topic = self._sample_index(
                        probabilities,
                        rng,
                    )

                    assignments[token_index] = new_topic
                    document_topic[new_topic] += 1

            topic_distribution = (
                document_topic + self.alpha
            ) / (
                len(sequence)
                + self.n_topics * self.alpha
            )

            output_scores[document_index] = (
                topic_distribution
                @ self.label_given_topic
            )

        return output_scores


class DenseMultiLabelDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(
            np.asarray(x, dtype=np.float32),
            dtype=torch.float32,
        )
        self.y = torch.tensor(
            np.asarray(y, dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class SetMultiLabelDataset(Dataset):
    def __init__(self, sequences, y):
        self.sequences = [
            [int(index) + 1 for index in sequence]
            for sequence in sequences
        ]
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.y[index]


def build_set_collate_function(cls_id):
    cls_id = int(cls_id)

    def collate(batch):
        sequences, labels = zip(*batch)

        lengths = [
            len(sequence) + 1
            for sequence in sequences
        ]
        max_length = max(lengths)

        token_ids = torch.zeros(
            (len(batch), max_length),
            dtype=torch.long,
        )
        padding_mask = torch.ones(
            (len(batch), max_length),
            dtype=torch.bool,
        )

        for row_index, sequence in enumerate(sequences):
            full_sequence = [cls_id] + list(sequence)
            length = len(full_sequence)

            token_ids[row_index, :length] = torch.tensor(
                full_sequence,
                dtype=torch.long,
            )
            padding_mask[row_index, :length] = False

        label_tensor = torch.tensor(
            np.stack(labels),
            dtype=torch.float32,
        )

        return token_ids, padding_mask, label_tensor

    return collate


class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        hidden_size2=128,
        dropout=0.15,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size2),
            nn.LayerNorm(hidden_size2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, x):
        return self.network(x)


class SetTransformerBaseline(nn.Module):
    def __init__(
        self,
        symptom_count,
        output_size,
        embedding_dim=128,
        n_heads=4,
        n_layers=2,
        feedforward_dim=256,
        dropout=0.15,
    ):
        super().__init__()

        if embedding_dim % n_heads != 0:
            raise ValueError(
                "embedding_dim 必须能被 n_heads 整除"
            )

        self.symptom_count = int(symptom_count)
        self.cls_id = self.symptom_count + 1
        self.vocabulary_size = self.symptom_count + 2

        self.embedding = nn.Embedding(
            self.vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, output_size),
        )

    def forward(self, token_ids, padding_mask):
        embeddings = self.embedding(token_ids)

        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=padding_mask,
        )

        cls_representation = encoded[:, 0]
        return self.classifier(cls_representation)


def compute_pos_weight(y_train, maximum=8.0):
    y_train = np.asarray(y_train, dtype=np.float64)

    counts = np.ones(
        y_train.shape[1],
        dtype=np.float64,
    )
    counts += y_train.sum(axis=0)

    negatives = np.maximum(
        len(y_train) - counts,
        1.0,
    )

    pos_weight = np.sqrt(
        negatives / np.maximum(counts, 1.0)
    )
    pos_weight = np.clip(
        pos_weight,
        1.0,
        float(maximum),
    )

    return torch.tensor(
        pos_weight,
        dtype=torch.float32,
    )


def autocast_context(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()

    if hasattr(torch, "autocast"):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return torch.cuda.amp.autocast(dtype=torch.float16)


def create_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )

    return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_step(
    loss,
    optimizer,
    model,
    scaler,
    amp_enabled,
    grad_clip=5.0,
):
    optimizer.zero_grad(set_to_none=True)

    if amp_enabled:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(grad_clip),
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(grad_clip),
        )
        optimizer.step()


def predict_mlp_scores(model, x, device, batch_size):
    model.eval()

    x_tensor = torch.tensor(
        np.asarray(x, dtype=np.float32),
        dtype=torch.float32,
    )

    loader = DataLoader(
        x_tensor,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )

    output = []

    with torch.no_grad():
        for features in loader:
            features = features.to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = model(features)
            output.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    return np.concatenate(output, axis=0)


def predict_set_transformer_scores(
    model,
    sequences,
    y_dummy,
    device,
    batch_size,
):
    model.eval()

    dataset = SetMultiLabelDataset(
        sequences,
        y_dummy,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=build_set_collate_function(model.cls_id),
    )

    output = []

    with torch.no_grad():
        for token_ids, padding_mask, _ in loader:
            token_ids = token_ids.to(
                device,
                non_blocking=device.type == "cuda",
            )
            padding_mask = padding_mask.to(
                device,
                non_blocking=device.type == "cuda",
            )

            logits = model(token_ids, padding_mask)
            output.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    return np.concatenate(output, axis=0)


def train_neural_baseline(
    model_name,
    data_bundle,
    args,
    seed,
    output_dir,
    device,
    logger,
):
    set_all_seeds(seed)

    train_data = data_bundle["encoded"]["train"]
    validation_data = data_bundle["encoded"]["validation"]
    test_data = data_bundle["encoded"]["test"]

    label_names = data_bundle["label_names"]
    train_support = data_bundle["train_support"]

    y_train = train_data["y"]
    y_validation = validation_data["y"]
    y_test = test_data["y"]

    if model_name in ("mlp_bce", "mlp_cb"):
        x_train = train_data["x_sparse"].toarray().astype(np.float32)
        x_validation = validation_data["x_sparse"].toarray().astype(np.float32)
        x_test = test_data["x_sparse"].toarray().astype(np.float32)

        model = MLPBaseline(
            input_size=x_train.shape[1],
            output_size=y_train.shape[1],
            hidden_size=args.mlp_hidden,
            hidden_size2=args.mlp_hidden2,
            dropout=args.dropout,
        ).to(device)

        train_dataset = DenseMultiLabelDataset(
            x_train,
            y_train,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
            pin_memory=device.type == "cuda",
        )

        def predict_validation():
            return predict_mlp_scores(
                model,
                x_validation,
                device,
                args.eval_batch_size,
            )

        def predict_test():
            return predict_mlp_scores(
                model,
                x_test,
                device,
                args.eval_batch_size,
            )

        def forward_batch(batch):
            features, targets = batch

            features = features.to(
                device,
                non_blocking=device.type == "cuda",
            )
            targets = targets.to(
                device,
                non_blocking=device.type == "cuda",
            )

            logits = model(features)
            return logits, targets

        model_config = {
            "input_size": int(x_train.shape[1]),
            "output_size": int(y_train.shape[1]),
            "hidden_size": int(args.mlp_hidden),
            "hidden_size2": int(args.mlp_hidden2),
            "dropout": float(args.dropout),
        }

    elif model_name == "set_transformer":
        symptom_count = len(data_bundle["symptom_to_index"])

        model = SetTransformerBaseline(
            symptom_count=symptom_count,
            output_size=y_train.shape[1],
            embedding_dim=args.transformer_dim,
            n_heads=args.transformer_heads,
            n_layers=args.transformer_layers,
            feedforward_dim=args.transformer_ffn,
            dropout=args.dropout,
        ).to(device)

        train_dataset = SetMultiLabelDataset(
            train_data["sequences"],
            y_train,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
            pin_memory=device.type == "cuda",
            collate_fn=build_set_collate_function(model.cls_id),
        )

        def predict_validation():
            return predict_set_transformer_scores(
                model,
                validation_data["sequences"],
                y_validation,
                device,
                args.eval_batch_size,
            )

        def predict_test():
            return predict_set_transformer_scores(
                model,
                test_data["sequences"],
                y_test,
                device,
                args.eval_batch_size,
            )

        def forward_batch(batch):
            token_ids, padding_mask, targets = batch

            token_ids = token_ids.to(
                device,
                non_blocking=device.type == "cuda",
            )
            padding_mask = padding_mask.to(
                device,
                non_blocking=device.type == "cuda",
            )
            targets = targets.to(
                device,
                non_blocking=device.type == "cuda",
            )

            logits = model(token_ids, padding_mask)
            return logits, targets

        model_config = {
            "symptom_count": int(symptom_count),
            "output_size": int(y_train.shape[1]),
            "embedding_dim": int(args.transformer_dim),
            "n_heads": int(args.transformer_heads),
            "n_layers": int(args.transformer_layers),
            "feedforward_dim": int(args.transformer_ffn),
            "dropout": float(args.dropout),
            "position_encoding": False,
        }

    else:
        raise ValueError(f"未知神经网络模型：{model_name}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    if model_name == "mlp_bce":
        criterion = nn.BCEWithLogitsLoss()
        loss_name = "bce"
        pos_weight = None
    else:
        pos_weight = compute_pos_weight(
            y_train,
            maximum=args.pos_weight_max,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )
        loss_name = "class_balanced_bce"

    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = create_grad_scaler(amp_enabled)

    thresholds = np.asarray(
        args.threshold_values,
        dtype=np.float64,
    )

    best_state = None
    best_epoch = -1
    best_threshold = None
    best_validation_result = None
    best_key = None
    history = []

    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            with autocast_context(device, amp_enabled):
                logits, targets = forward_batch(batch)
                loss = criterion(logits, targets)

            optimizer_step(
                loss,
                optimizer,
                model,
                scaler,
                amp_enabled,
                grad_clip=args.grad_clip,
            )

            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        validation_scores = predict_validation()

        threshold_result, threshold_history = choose_best_threshold(
            validation_scores,
            y_validation,
            label_names,
            train_support,
            thresholds,
            min_labels=args.min_labels,
        )

        validation_result = threshold_result["metrics"]
        validation_key = selection_key(
            validation_result,
            threshold_result["threshold"],
        )

        improved = best_key is None or validation_key > best_key

        if improved:
            best_key = validation_key
            best_epoch = epoch
            best_threshold = threshold_result["threshold"]
            best_validation_result = validation_result
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        epoch_row = {
            "epoch": int(epoch),
            "train_loss": (
                total_loss / total_samples
                if total_samples > 0
                else 0.0
            ),
            "threshold": float(threshold_result["threshold"]),
            "improved": bool(improved),
            **validation_result["auto"],
        }
        history.append(epoch_row)

        logger.info(
            f"{model_name} seed={seed} "
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={epoch_row['train_loss']:.6f} "
            f"threshold={epoch_row['threshold']:.4f} "
            f"val_supported_macro_f1="
            f"{epoch_row['supported_macro_f1']:.4f} "
            f"val_sample_f1={epoch_row['sample_f1']:.4f} "
            f"best_epoch={best_epoch + 1}"
        )

    if best_state is None:
        raise RuntimeError(f"{model_name} 未产生最佳模型")

    model.load_state_dict(best_state)

    validation_scores = predict_validation()
    validation_predictions = threshold_predictions(
        validation_scores,
        best_threshold,
        min_labels=args.min_labels,
    )
    validation_result = evaluate_binary_predictions(
        y_validation,
        validation_predictions,
        label_names,
        train_support,
    )

    test_scores = predict_test()
    test_predictions = threshold_predictions(
        test_scores,
        best_threshold,
        min_labels=args.min_labels,
    )
    test_result = evaluate_binary_predictions(
        y_test,
        test_predictions,
        label_names,
        train_support,
    )

    elapsed = time.perf_counter() - start_time

    checkpoint = {
        "checkpoint_version": 1,
        "model_name": model_name,
        "display_name": MODEL_DISPLAY_NAMES[model_name],
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "model_config": model_config,
        "loss_name": loss_name,
        "pos_weight": (
            pos_weight.detach().cpu()
            if pos_weight is not None
            else None
        ),
        "state_dict": best_state,
        "label_names": label_names,
        "symptom_to_index": data_bundle["symptom_to_index"],
        "split_manifest": data_bundle["manifest"],
    }

    torch.save(
        checkpoint,
        output_dir / "best.pt",
    )

    write_json(
        output_dir / "history.json",
        history,
    )

    write_json(
        output_dir / "resolved_config.json",
        {
            "model_name": model_name,
            "display_name": MODEL_DISPLAY_NAMES[model_name],
            "seed": seed,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "loss_name": loss_name,
            "model_config": model_config,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pos_weight_max": args.pos_weight_max,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "elapsed_seconds": elapsed,
        },
    )

    save_evaluation_artifacts(
        output_dir,
        "validation",
        validation_data,
        validation_scores,
        validation_predictions,
        validation_result,
        label_names,
    )

    save_evaluation_artifacts(
        output_dir,
        "test",
        test_data,
        test_scores,
        test_predictions,
        test_result,
        label_names,
    )

    return {
        "model": model_name,
        "display_name": MODEL_DISPLAY_NAMES[model_name],
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "elapsed_seconds": float(elapsed),
        "validation": validation_result,
        "test": test_result,
    }


def run_frequency_baseline(
    data_bundle,
    args,
    seed,
    output_dir,
    logger,
):
    set_all_seeds(seed)

    train_data = data_bundle["encoded"]["train"]
    validation_data = data_bundle["encoded"]["validation"]
    test_data = data_bundle["encoded"]["test"]

    label_names = data_bundle["label_names"]
    train_support = data_bundle["train_support"]

    model = FrequencyBaseline().fit(train_data["y"])

    validation_scores = model.predict_scores(
        len(validation_data["y"])
    )
    test_scores = model.predict_scores(
        len(test_data["y"])
    )

    max_k = min(
        args.frequency_max_k,
        len(label_names),
    )

    k_values = list(range(1, max_k + 1))

    best_k_result, k_history = choose_best_k(
        validation_scores,
        validation_data["y"],
        label_names,
        train_support,
        k_values,
    )

    best_k = best_k_result["k"]

    validation_predictions = top_k_predictions(
        validation_scores,
        best_k,
    )
    test_predictions = top_k_predictions(
        test_scores,
        best_k,
    )

    validation_result = evaluate_binary_predictions(
        validation_data["y"],
        validation_predictions,
        label_names,
        train_support,
    )
    test_result = evaluate_binary_predictions(
        test_data["y"],
        test_predictions,
        label_names,
        train_support,
    )

    with (output_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump({
            "model_name": "frequency",
            "prevalence": model.prevalence,
            "best_k": best_k,
            "label_names": label_names,
        }, file_obj)

    write_json(
        output_dir / "validation_search.json",
        k_history,
    )

    write_json(
        output_dir / "resolved_config.json",
        {
            "model_name": "frequency",
            "display_name": MODEL_DISPLAY_NAMES["frequency"],
            "seed": seed,
            "best_k": best_k,
            "prevalence": model.prevalence,
        },
    )

    save_evaluation_artifacts(
        output_dir,
        "validation",
        validation_data,
        validation_scores,
        validation_predictions,
        validation_result,
        label_names,
    )

    save_evaluation_artifacts(
        output_dir,
        "test",
        test_data,
        test_scores,
        test_predictions,
        test_result,
        label_names,
    )

    logger.info(
        f"frequency seed={seed} best_k={best_k} "
        f"test_supported_macro_f1="
        f"{test_result['auto']['supported_macro_f1']:.4f}"
    )

    return {
        "model": "frequency",
        "display_name": MODEL_DISPLAY_NAMES["frequency"],
        "seed": int(seed),
        "best_k": int(best_k),
        "validation": validation_result,
        "test": test_result,
    }


def run_svm_baseline(
    data_bundle,
    args,
    seed,
    output_dir,
    logger,
):
    set_all_seeds(seed)

    train_data = data_bundle["encoded"]["train"]
    validation_data = data_bundle["encoded"]["validation"]
    test_data = data_bundle["encoded"]["test"]

    label_names = data_bundle["label_names"]
    train_support = data_bundle["train_support"]

    thresholds = np.asarray(
        args.threshold_values,
        dtype=np.float64,
    )

    best = None
    search_rows = []
    start_time = time.perf_counter()

    for c_value in args.svm_c_values:
        model = BinaryRelevanceLinearSVM(
            c=c_value,
            seed=seed,
        ).fit(
            train_data["x_sparse"],
            train_data["y"],
        )

        validation_scores = model.predict_scores(
            validation_data["x_sparse"]
        )

        threshold_result, threshold_history = choose_best_threshold(
            validation_scores,
            validation_data["y"],
            label_names,
            train_support,
            thresholds,
            min_labels=args.min_labels,
        )

        validation_result = threshold_result["metrics"]

        for row in threshold_history:
            search_rows.append({
                "c": float(c_value),
                **row,
            })

        candidate = {
            "model": model,
            "c": float(c_value),
            "threshold": float(threshold_result["threshold"]),
            "validation_scores": validation_scores,
            "validation_predictions": threshold_result["predictions"],
            "validation_result": validation_result,
            "key": selection_key(
                validation_result,
                threshold_result["threshold"],
            ),
        }

        if best is None or candidate["key"] > best["key"]:
            best = candidate

    model = best["model"]
    best_threshold = best["threshold"]

    validation_scores = model.predict_scores(
        validation_data["x_sparse"]
    )
    validation_predictions = threshold_predictions(
        validation_scores,
        best_threshold,
        min_labels=args.min_labels,
    )
    validation_result = evaluate_binary_predictions(
        validation_data["y"],
        validation_predictions,
        label_names,
        train_support,
    )

    test_scores = model.predict_scores(
        test_data["x_sparse"]
    )
    test_predictions = threshold_predictions(
        test_scores,
        best_threshold,
        min_labels=args.min_labels,
    )
    test_result = evaluate_binary_predictions(
        test_data["y"],
        test_predictions,
        label_names,
        train_support,
    )

    elapsed = time.perf_counter() - start_time

    with (output_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump({
            "model_name": "br_svm",
            "model": model,
            "best_c": best["c"],
            "best_threshold": best_threshold,
            "label_names": label_names,
            "symptom_to_index": data_bundle["symptom_to_index"],
        }, file_obj)

    write_csv(
        output_dir / "validation_search.csv",
        search_rows,
    )

    write_json(
        output_dir / "resolved_config.json",
        {
            "model_name": "br_svm",
            "display_name": MODEL_DISPLAY_NAMES["br_svm"],
            "seed": seed,
            "best_c": best["c"],
            "best_threshold": best_threshold,
            "c_values": args.svm_c_values,
            "elapsed_seconds": elapsed,
        },
    )

    save_evaluation_artifacts(
        output_dir,
        "validation",
        validation_data,
        validation_scores,
        validation_predictions,
        validation_result,
        label_names,
    )

    save_evaluation_artifacts(
        output_dir,
        "test",
        test_data,
        test_scores,
        test_predictions,
        test_result,
        label_names,
    )

    logger.info(
        f"br_svm seed={seed} C={best['c']} "
        f"threshold={best_threshold:.4f} "
        f"test_supported_macro_f1="
        f"{test_result['auto']['supported_macro_f1']:.4f}"
    )

    return {
        "model": "br_svm",
        "display_name": MODEL_DISPLAY_NAMES["br_svm"],
        "seed": int(seed),
        "best_c": float(best["c"]),
        "best_threshold": float(best_threshold),
        "elapsed_seconds": float(elapsed),
        "validation": validation_result,
        "test": test_result,
    }


def run_mlknn_baseline(
    data_bundle,
    args,
    seed,
    output_dir,
    logger,
):
    set_all_seeds(seed)

    train_data = data_bundle["encoded"]["train"]
    validation_data = data_bundle["encoded"]["validation"]
    test_data = data_bundle["encoded"]["test"]

    label_names = data_bundle["label_names"]
    train_support = data_bundle["train_support"]

    thresholds = np.asarray(
        args.threshold_values,
        dtype=np.float64,
    )

    best = None
    search_rows = []
    start_time = time.perf_counter()

    for k in args.mlknn_k_values:
        for smoothing in args.mlknn_smoothing_values:
            model = MLKNN(
                n_neighbors=k,
                smoothing=smoothing,
                metric=args.mlknn_metric,
                n_jobs=args.n_jobs,
            ).fit(
                train_data["x_sparse"],
                train_data["y"],
            )

            validation_scores = model.predict_scores(
                validation_data["x_sparse"]
            )

            threshold_result, threshold_history = choose_best_threshold(
                validation_scores,
                validation_data["y"],
                label_names,
                train_support,
                thresholds,
                min_labels=args.min_labels,
            )

            validation_result = threshold_result["metrics"]

            for row in threshold_history:
                search_rows.append({
                    "k": int(k),
                    "effective_k": int(model.effective_k),
                    "smoothing": float(smoothing),
                    "metric": args.mlknn_metric,
                    **row,
                })

            candidate = {
                "model": model,
                "k": int(k),
                "effective_k": int(model.effective_k),
                "smoothing": float(smoothing),
                "threshold": float(threshold_result["threshold"]),
                "validation_result": validation_result,
                "key": selection_key(
                    validation_result,
                    threshold_result["threshold"],
                ),
            }

            if best is None or candidate["key"] > best["key"]:
                best = candidate

    model = best["model"]
    best_threshold = best["threshold"]

    validation_scores = model.predict_scores(
        validation_data["x_sparse"]
    )
    validation_predictions = threshold_predictions(
        validation_scores,
        best_threshold,
        min_labels=args.min_labels,
    )
    validation_result = evaluate_binary_predictions(
        validation_data["y"],
        validation_predictions,
        label_names,
        train_support,
    )

    test_scores = model.predict_scores(
        test_data["x_sparse"]
    )
    test_predictions = threshold_predictions(
        test_scores,
        best_threshold,
        min_labels=args.min_labels,
    )
    test_result = evaluate_binary_predictions(
        test_data["y"],
        test_predictions,
        label_names,
        train_support,
    )

    elapsed = time.perf_counter() - start_time

    with (output_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump({
            "model_name": "mlknn",
            "model": model,
            "best_k": best["k"],
            "effective_k": best["effective_k"],
            "best_smoothing": best["smoothing"],
            "best_threshold": best_threshold,
            "label_names": label_names,
            "symptom_to_index": data_bundle["symptom_to_index"],
        }, file_obj)

    write_csv(
        output_dir / "validation_search.csv",
        search_rows,
    )

    write_json(
        output_dir / "resolved_config.json",
        {
            "model_name": "mlknn",
            "display_name": MODEL_DISPLAY_NAMES["mlknn"],
            "seed": seed,
            "best_k": best["k"],
            "effective_k": best["effective_k"],
            "best_smoothing": best["smoothing"],
            "best_threshold": best_threshold,
            "metric": args.mlknn_metric,
            "k_values": args.mlknn_k_values,
            "smoothing_values": args.mlknn_smoothing_values,
            "elapsed_seconds": elapsed,
        },
    )

    save_evaluation_artifacts(
        output_dir,
        "validation",
        validation_data,
        validation_scores,
        validation_predictions,
        validation_result,
        label_names,
    )

    save_evaluation_artifacts(
        output_dir,
        "test",
        test_data,
        test_scores,
        test_predictions,
        test_result,
        label_names,
    )

    logger.info(
        f"mlknn seed={seed} k={best['effective_k']} "
        f"smoothing={best['smoothing']} "
        f"threshold={best_threshold:.4f} "
        f"test_supported_macro_f1="
        f"{test_result['auto']['supported_macro_f1']:.4f}"
    )

    return {
        "model": "mlknn",
        "display_name": MODEL_DISPLAY_NAMES["mlknn"],
        "seed": int(seed),
        "best_k": int(best["k"]),
        "effective_k": int(best["effective_k"]),
        "best_smoothing": float(best["smoothing"]),
        "best_threshold": float(best_threshold),
        "elapsed_seconds": float(elapsed),
        "validation": validation_result,
        "test": test_result,
    }


def run_stacv2_baseline(
    data_bundle,
    args,
    seed,
    output_dir,
    logger,
):
    """运行StaCv2 + Logistic Regression基线。"""
    set_all_seeds(seed)

    train_data = data_bundle["encoded"]["train"]
    validation_data = data_bundle["encoded"]["validation"]
    test_data = data_bundle["encoded"]["test"]

    label_names = data_bundle["label_names"]
    train_support = data_bundle["train_support"]

    thresholds = np.asarray(
        args.threshold_values,
        dtype=np.float64,
    )

    start_time = time.perf_counter()

    model = StaCv2(
        c=args.stac_c,
        n_folds=args.stac_folds,
        internal_threshold=args.stac_internal_threshold,
        max_iter=args.stac_max_iter,
        seed=seed,
    )

    logger.info(
        f"开始训练StaCv2：seed={seed}, "
        f"C={args.stac_c}, "
        f"folds={args.stac_folds}, "
        f"internal_threshold="
        f"{args.stac_internal_threshold}"
    )

    model.fit(
        train_data["x_sparse"],
        train_data["y"],
    )

    # ---------------------------------------------------------
    # 在验证集上选择最终输出阈值。
    #
    # internal_threshold仅用于链内部的标签状态更新；
    # best_threshold用于将第二层最终概率转换成最终预测集合。
    # ---------------------------------------------------------
    validation_scores = model.predict_scores(
        validation_data["x_sparse"]
    )

    threshold_result, threshold_history = choose_best_threshold(
        validation_scores,
        validation_data["y"],
        label_names,
        train_support,
        thresholds,
        min_labels=args.min_labels,
    )

    best_threshold = float(
        threshold_result["threshold"]
    )

    validation_predictions = threshold_predictions(
        validation_scores,
        best_threshold,
        min_labels=args.min_labels,
    )

    validation_result = evaluate_binary_predictions(
        validation_data["y"],
        validation_predictions,
        label_names,
        train_support,
    )

    # ---------------------------------------------------------
    # 测试集评估。测试集不能重新调整阈值。
    # ---------------------------------------------------------
    test_scores = model.predict_scores(
        test_data["x_sparse"]
    )

    test_predictions = threshold_predictions(
        test_scores,
        best_threshold,
        min_labels=args.min_labels,
    )

    test_result = evaluate_binary_predictions(
        test_data["y"],
        test_predictions,
        label_names,
        train_support,
    )

    elapsed = time.perf_counter() - start_time

    # 保存模型。
    with (output_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump({
            "model_name": "stacv2",
            "display_name": MODEL_DISPLAY_NAMES["stacv2"],
            "model": model,
            "best_threshold": best_threshold,
            "label_order": model.label_order,
            "label_names": label_names,
            "symptom_to_index": (
                data_bundle["symptom_to_index"]
            ),
            "split_manifest": data_bundle["manifest"],
        }, file_obj)

    write_csv(
        output_dir / "validation_search.csv",
        threshold_history,
    )

    write_json(
        output_dir / "resolved_config.json",
        {
            "model_name": "stacv2",
            "display_name": (
                MODEL_DISPLAY_NAMES["stacv2"]
            ),
            "seed": int(seed),
            "base_classifier": (
                "LogisticRegression"
            ),
            "solver": "lbfgs",
            "penalty": "l2",
            "c": float(args.stac_c),
            "tol": 1e-4,
            "max_iter": int(args.stac_max_iter),
            "oof_folds": int(args.stac_folds),
            "internal_threshold": float(
                args.stac_internal_threshold
            ),
            "best_output_threshold": float(
                best_threshold
            ),
            "label_order_indices": (
                model.label_order
            ),
            "label_order_names": [
                label_names[index]
                for index in model.label_order
            ],
            "elapsed_seconds": float(elapsed),
        },
    )

    save_evaluation_artifacts(
        output_dir,
        "validation",
        validation_data,
        validation_scores,
        validation_predictions,
        validation_result,
        label_names,
    )

    save_evaluation_artifacts(
        output_dir,
        "test",
        test_data,
        test_scores,
        test_predictions,
        test_result,
        label_names,
    )

    logger.info(
        f"stacv2 seed={seed} "
        f"threshold={best_threshold:.4f} "
        f"test_sample_f1="
        f"{test_result['auto']['sample_f1']:.4f} "
        f"test_micro_f1="
        f"{test_result['auto']['micro_f1']:.4f} "
        f"test_supported_macro_f1="
        f"{test_result['auto']['supported_macro_f1']:.4f} "
        f"test_exact_match="
        f"{test_result['auto']['exact_match']:.4f} "
        f"elapsed={elapsed:.2f}s"
    )

    return {
        "model": "stacv2",
        "display_name": MODEL_DISPLAY_NAMES["stacv2"],
        "seed": int(seed),
        "best_threshold": float(best_threshold),
        "elapsed_seconds": float(elapsed),
        "label_order": model.label_order,
        "validation": validation_result,
        "test": test_result,
    }

def run_ptm_top3_baseline(
    data_bundle,
    args,
    seed,
    output_dir,
    logger,
):
    set_all_seeds(seed)

    train_data = data_bundle["encoded"]["train"]
    validation_data = data_bundle["encoded"]["validation"]
    test_data = data_bundle["encoded"]["test"]

    label_names = data_bundle["label_names"]
    train_support = data_bundle["train_support"]
    symptom_count = len(
        data_bundle["symptom_to_index"]
    )

    start_time = time.perf_counter()

    model = PTMFixedTopK(
        n_topics=args.ptm_topics,
        n_roles=args.ptm_roles,
        alpha=args.ptm_alpha,
        beta_symptom=args.ptm_beta,
        beta_label=args.ptm_beta,
        eta=args.ptm_eta,
        iterations=args.ptm_iterations,
        inference_iterations=args.ptm_inference_iterations,
        seed=seed,
    )

    model.fit(
        train_data["sequences"],
        train_data["y"],
        symptom_count=symptom_count,
    )

    validation_scores = model.predict_scores(
        validation_data["sequences"],
        seed_offset=1,
    )
    validation_predictions = top_k_predictions(
        validation_scores,
        k=3,
    )
    validation_result = evaluate_binary_predictions(
        validation_data["y"],
        validation_predictions,
        label_names,
        train_support,
    )

    test_scores = model.predict_scores(
        test_data["sequences"],
        seed_offset=2,
    )
    test_predictions = top_k_predictions(
        test_scores,
        k=3,
    )
    test_result = evaluate_binary_predictions(
        test_data["y"],
        test_predictions,
        label_names,
        train_support,
    )

    elapsed = time.perf_counter() - start_time

    with (output_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump({
            "model_name": "ptm_top3",
            "model": model,
            "fixed_output_count": 3,
            "label_names": label_names,
            "symptom_to_index": (
                data_bundle["symptom_to_index"]
            ),
        }, file_obj)

    write_json(
        output_dir / "resolved_config.json",
        {
            "model_name": "ptm_top3",
            "display_name": (
                MODEL_DISPLAY_NAMES["ptm_top3"]
            ),
            "seed": int(seed),
            "topics": int(args.ptm_topics),
            "roles": int(args.ptm_roles),
            "alpha": float(args.ptm_alpha),
            "beta": float(args.ptm_beta),
            "eta": float(args.ptm_eta),
            "iterations": int(args.ptm_iterations),
            "inference_iterations": int(
                args.ptm_inference_iterations
            ),
            "fixed_output_count": 3,
            "elapsed_seconds": float(elapsed),
        },
    )

    save_evaluation_artifacts(
        output_dir,
        "validation",
        validation_data,
        validation_scores,
        validation_predictions,
        validation_result,
        label_names,
    )

    save_evaluation_artifacts(
        output_dir,
        "test",
        test_data,
        test_scores,
        test_predictions,
        test_result,
        label_names,
    )

    logger.info(
        f"ptm_top3 seed={seed} "
        f"topics={args.ptm_topics} "
        f"test_sample_f1="
        f"{test_result['auto']['sample_f1']:.4f} "
        f"test_supported_macro_f1="
        f"{test_result['auto']['supported_macro_f1']:.4f} "
        f"test_exact_match="
        f"{test_result['auto']['exact_match']:.4f}"
    )

    return {
        "model": "ptm_top3",
        "display_name": MODEL_DISPLAY_NAMES["ptm_top3"],
        "seed": int(seed),
        "topics": int(args.ptm_topics),
        "fixed_k": 3,
        "elapsed_seconds": float(elapsed),
        "validation": validation_result,
        "test": test_result,
    }


def run_model(
    model_name,
    data_bundle,
    args,
    seed,
    output_dir,
    device,
    logger,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "frequency":
        return run_frequency_baseline(
            data_bundle,
            args,
            seed,
            output_dir,
            logger,
        )

    if model_name == "br_svm":
        return run_svm_baseline(
            data_bundle,
            args,
            seed,
            output_dir,
            logger,
        )

    if model_name == "mlknn":
        return run_mlknn_baseline(
            data_bundle,
            args,
            seed,
            output_dir,
            logger,
        )
        
    if model_name == "ptm_top3":
        return run_ptm_top3_baseline(
            data_bundle,
            args,
            seed,
            output_dir,
            logger,
        )


    if model_name == "stacv2":
        return run_stacv2_baseline(
            data_bundle,
            args,
            seed,
            output_dir,
            logger,
        )

    if model_name in (
        "mlp_bce",
        "mlp_cb",
        "set_transformer",
    ):

        return train_neural_baseline(
            model_name,
            data_bundle,
            args,
            seed,
            output_dir,
            device,
            logger,
        )

    raise ValueError(f"不支持的模型：{model_name}")


def build_per_seed_rows(results):
    rows = []

    for result in results:
        row = {
            "model": result["model"],
            "display_name": result["display_name"],
            "seed": result["seed"],
        }

        for metric_name in SUMMARY_METRICS:
            row[metric_name] = result["test"]["auto"].get(
                metric_name,
                0.0,
            )

        for key in (
            "best_epoch",
            "best_threshold",
            "best_k",
            "effective_k",
            "best_c",
            "best_smoothing",
            "topics",
            "fixed_k",
            "elapsed_seconds",
        ):

            if key in result:
                row[key] = result[key]

        rows.append(row)

    return rows


def aggregate_results(results, selected_models):
    grouped = {
        model_name: []
        for model_name in selected_models
    }

    for result in results:
        grouped[result["model"]].append(result)

    summary_rows = []
    summary_json = {}

    for model_name in selected_models:
        model_results = grouped.get(model_name, [])

        if not model_results:
            continue

        summary_row = {
            "model": model_name,
            "display_name": MODEL_DISPLAY_NAMES[model_name],
            "seed_count": len(model_results),
        }

        summary_json[model_name] = {
            "display_name": MODEL_DISPLAY_NAMES[model_name],
            "seed_count": len(model_results),
            "metrics": {},
        }

        for metric_name in SUMMARY_METRICS:
            values = np.asarray([
                result["test"]["auto"].get(metric_name, 0.0)
                for result in model_results
            ], dtype=np.float64)

            mean_value = float(np.mean(values))
            std_value = float(np.std(values))

            summary_row[f"{metric_name}_mean"] = mean_value
            summary_row[f"{metric_name}_std"] = std_value

            summary_json[model_name]["metrics"][metric_name] = {
                "mean": mean_value,
                "std": std_value,
                "values": values.tolist(),
            }

        summary_rows.append(summary_row)

    return summary_rows, summary_json


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    output = str(text)

    for source, target in replacements.items():
        output = output.replace(source, target)

    return output


def format_mean_std(row, metric, digits=4):
    mean_value = float(row[f"{metric}_mean"])
    std_value = float(row[f"{metric}_std"])
    return f"{mean_value:.{digits}f} $\\pm$ {std_value:.{digits}f}"


def write_latex_table(path, summary_rows):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Comparison of the proposed method with "
            r"baseline methods on the test set. Higher values "
            r"indicate better performance, except for Miss Rate "
            r"and Zero-F1 Labels.}"
        ),
        r"\label{tab:main_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\hline",
        (
            r"Method & Sample F1 & Micro-F1 & Supported Macro-F1 "
            r"& Exact Match & Sample Precision & Sample Recall "
            r"& Miss Rate & Zero-F1 Labels \\"
        ),
        r"\hline",
    ]


    for row in summary_rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["display_name"]),
                format_mean_std(row, "sample_f1"),
                format_mean_std(row, "micro_f1"),
                format_mean_std(row, "supported_macro_f1"),
                format_mean_std(row, "exact_match"),
                format_mean_std(row, "sample_precision"),
                format_mean_std(row, "sample_recall"),
                format_mean_std(row, "miss_selection_rate"),
                format_mean_std(row, "zero_f1_count", digits=2),
            )
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])

    Path(path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_number_list(text, cast_type):
    return [
        cast_type(value.strip())
        for value in str(text).split(",")
        if value.strip()
    ]


def resolve_models(text):
    text = str(text).strip().lower()

    if text == "all":
        return list(MODEL_NAMES)

    models = [
        value.strip()
        for value in text.split(",")
        if value.strip()
    ]

    unknown = [
        model_name
        for model_name in models
        if model_name not in MODEL_NAMES
    ]

    if unknown:
        raise ValueError(
            f"未知模型：{unknown}；支持：{MODEL_NAMES}"
        )

    return models


def default_data_path():
    project_dataset = PROJECT_ROOT / "dataset" / "lhz_data_cleaned.txt"
    local_dataset = MYMODEL_DIR / "lhz_data_cleaned.txt"

    if project_dataset.exists():
        return str(project_dataset)

    return str(local_dataset)


def build_parser():
    parser = argparse.ArgumentParser(
        description="统一多标签基线实验"
    )

    parser.add_argument(
        "--data-path",
        default=default_data_path(),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--models",
        default="all",
    )
    parser.add_argument(
        "--seeds",
        default="9,17,29",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--max-se-num",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--unknown-label-policy",
        choices=("drop_sample", "error"),
        default="drop_sample",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cuda-tf32",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--min-labels",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--frequency-max-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--svm-c-values",
        default="0.1,1.0,10.0",
    )

    parser.add_argument(
        "--mlknn-k-values",
        default="5,10,15,20",
    )
    parser.add_argument(
        "--mlknn-smoothing-values",
        default="1.0",
    )
    parser.add_argument(
        "--mlknn-metric",
        choices=("cosine", "euclidean", "manhattan"),
        default="cosine",
    )
# StaCv2参数。
    parser.add_argument(
        "--stac-c",
        type=float,
        default=1.0,
        help=(
            "StaCv2中Logistic Regression的正则化参数C"
        ),
    )
    parser.add_argument(
        "--stac-folds",
        type=int,
        default=3,
        help=(
            "StaCv2生成训练集OOF预测时的交叉验证折数"
        ),
    )
    parser.add_argument(
        "--stac-internal-threshold",
        type=float,
        default=0.5,
        help=(
            "StaCv2链内部将概率转换为标签状态的阈值；"
            "该参数不同于最终输出阈值"
        ),
    )
    parser.add_argument(
        "--stac-max-iter",
        type=int,
        default=100,
        help=(
            "StaCv2中Logistic Regression的最大迭代次数"
        ),
    )
    
    parser.add_argument(
        "--ptm-topics",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--ptm-roles",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--ptm-alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ptm-beta",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ptm-eta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ptm-iterations",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--ptm-inference-iterations",
        type=int,
        default=100,
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--pos-weight-max",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--mlp-hidden2",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--transformer-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--transformer-ffn",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser



def resolve_arguments(args):
    args.models = resolve_models(args.models)
    args.seeds = parse_number_list(args.seeds, int)
    args.svm_c_values = parse_number_list(
        args.svm_c_values,
        float,
    )
    args.mlknn_k_values = parse_number_list(
        args.mlknn_k_values,
        int,
    )
    args.mlknn_smoothing_values = parse_number_list(
        args.mlknn_smoothing_values,
        float,
    )

    if not args.seeds:
        raise ValueError("seeds 不能为空")

    if not args.models:
        raise ValueError("models 不能为空")

    threshold_count = int(math.floor(
        (args.threshold_max - args.threshold_min)
        / args.threshold_step
    )) + 1

    thresholds = [
        args.threshold_min + index * args.threshold_step
        for index in range(threshold_count)
    ]

    thresholds.append(0.5)

    args.threshold_values = sorted({
        round(float(value), 10)
        for value in thresholds
        if 0.0 <= value <= 1.0
    })

    if args.transformer_dim % args.transformer_heads != 0:
        raise ValueError(
            "transformer-dim 必须能被 transformer-heads 整除"
        )

    ratio_sum = (
        args.train_ratio
        + args.val_ratio
        + args.test_ratio
    )

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "train-ratio、val-ratio、test-ratio 总和必须为1"
        )

    if args.stac_folds < 2:
        raise ValueError(
            "stac-folds必须不小于2"
        )

    if args.stac_c <= 0:
        raise ValueError(
            "stac-c必须大于0"
        )

    if args.stac_max_iter <= 0:
        raise ValueError(
            "stac-max-iter必须大于0"
        )

    if not (
        0.0
        <= args.stac_internal_threshold
        <= 1.0
    ):
        raise ValueError(
            "stac-internal-threshold必须位于[0, 1]"
        )
        
    if args.ptm_topics <= 0:
        raise ValueError("ptm-topics必须大于0")

    if args.ptm_roles <= 0:
        raise ValueError("ptm-roles必须大于0")

    if args.ptm_alpha <= 0:
        raise ValueError("ptm-alpha必须大于0")

    if args.ptm_beta <= 0:
        raise ValueError("ptm-beta必须大于0")

    if args.ptm_eta <= 0:
        raise ValueError("ptm-eta必须大于0")

    if args.ptm_iterations <= 0:
        raise ValueError("ptm-iterations必须大于0")

    if args.ptm_inference_iterations <= 0:
        raise ValueError(
            "ptm-inference-iterations必须大于0"
        )


    return args



def main():
    args = resolve_arguments(
        build_parser().parse_args()
    )

    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    else:
        output_root = (
            SCRIPT_DIR
            / "runs"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    output_root.mkdir(parents=True, exist_ok=True)

    logger = create_logger(
        output_root / "baseline.log"
    )

    device = resolve_device(args.device)
    accelerator = configure_accelerator(
        device,
        bool(args.cuda_tf32),
    )

    logger.info(
        f"device={device}, "
        f"gpu={accelerator['gpu_name']}, "
        f"CUDA={accelerator['cuda_version']}, "
        f"TF32={accelerator['tf32']}"
    )

    data_bundle = prepare_data(
        args,
        logger,
        output_root,
    )

    resolved_config = vars(args).copy()
    resolved_config.update({
        "data_path": str(Path(args.data_path).resolve()),
        "output_root": str(output_root),
        "resolved_device": str(device),
        "accelerator": accelerator,
        "symptom_count": len(data_bundle["symptom_to_index"]),
        "label_count": len(data_bundle["label_names"]),
        "train_count": len(data_bundle["encoded"]["train"]["y"]),
        "validation_count": len(
            data_bundle["encoded"]["validation"]["y"]
        ),
        "test_count": len(data_bundle["encoded"]["test"]["y"]),
        "split_manifest": data_bundle["manifest"],
    })

    write_json(
        output_root / "resolved_config.json",
        resolved_config,
    )

    if args.dry_run:
        logger.info("DRY RUN 完成")
        print(output_root)
        return

    all_results = []

    for model_name in args.models:
        for seed in args.seeds:
            seed_output_dir = (
                output_root
                / model_name
                / f"seed_{seed}"
            )

            logger.info(
                f"开始模型={model_name}, seed={seed}"
            )

            try:
                result = run_model(
                    model_name,
                    data_bundle,
                    args,
                    seed,
                    seed_output_dir,
                    device,
                    logger,
                )

                result["status"] = "done"
                all_results.append(result)

                write_json(
                    seed_output_dir / "run_result.json",
                    result,
                )

            except Exception as error:
                failure = {
                    "model": model_name,
                    "display_name": MODEL_DISPLAY_NAMES[model_name],
                    "seed": seed,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }

                all_results.append(failure)

                write_json(
                    seed_output_dir / "failure.json",
                    failure,
                )

                logger.exception(
                    f"模型运行失败：model={model_name}, seed={seed}"
                )

                if len(args.models) == 1 and len(args.seeds) == 1:
                    raise

    write_json(
        output_root / "all_results.json",
        all_results,
    )

    successful_results = [
        result
        for result in all_results
        if result.get("status") == "done"
    ]

    failed_results = [
        result
        for result in all_results
        if result.get("status") == "failed"
    ]

    per_seed_rows = build_per_seed_rows(
        successful_results
    )

    write_csv(
        output_root / "per_seed_results.csv",
        per_seed_rows,
    )

    write_json(
        output_root / "per_seed_results.json",
        per_seed_rows,
    )

    summary_rows, summary_json = aggregate_results(
        successful_results,
        args.models,
    )

    write_csv(
        output_root / "comparison_summary.csv",
        summary_rows,
    )

    write_json(
        output_root / "comparison_summary.json",
        summary_json,
    )

    write_latex_table(
        output_root / "comparison_table.tex",
        summary_rows,
    )

    write_json(
        output_root / "failed_runs.json",
        failed_results,
    )

    logger.info(
        f"全部实验完成：成功={len(successful_results)}, "
        f"失败={len(failed_results)}, "
        f"输出目录={output_root}"
    )

    print(output_root)


if __name__ == "__main__":
    main()
