import copy
import json
from pathlib import Path

import numpy as np


# ============================================================
# 配置
# ============================================================

# 当前Python脚本所在的目录
RUN_DIR = Path(__file__).resolve().parent

INPUT_PATH = RUN_DIR / "history.json"
OUTPUT_PATH = RUN_DIR / "history_demo_synthetic.json"
NOTICE_PATH = RUN_DIR / "DEMO_ONLY.txt"


# 明确使用demo命名，不覆盖原始数据
OUTPUT_PATH = RUN_DIR / "history_demo_synthetic.json"
NOTICE_PATH = RUN_DIR / "DEMO_ONLY.txt"

# 后期Recall目标，MR约为1-0.864=0.136
TARGET_RECALL = 0.864

# 第一轮附近的Recall起点
START_RECALL = 0.720

# 趋势上升速度，越小则越快进入稳定阶段
TAU = 5.0

# 后期震荡幅度
MAX_FLUCTUATION = 0.006

# # Supported Macro-F1的理想稳定值
# TARGET_SUPPORTED_MACRO_F1 = 0.30

# # 初始值，使曲线前期上升、后期在0.40附近震荡
# START_SUPPORTED_MACRO_F1 = 0.365

# # 收敛速度，越小则越快趋近目标值
# SUPPORTED_MACRO_TAU = 10.0

# # 后期最大波动幅度，通常约为±0.006
# SUPPORTED_MACRO_FLUCTUATION = 0.006


# 固定随机种子，使每次运行生成相同结果
RANDOM_SEED = 20260727


# ============================================================
# 工具函数
# ============================================================

def moving_average(values, window=7):
    """长度保持不变的中心移动平均。"""
    values = np.asarray(values, dtype=float)

    left = window // 2
    right = window - 1 - left

    padded = np.pad(
        values,
        (left, right),
        mode="edge",
    )

    kernel = np.ones(window, dtype=float) / window

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


# ============================================================
# 读取原始数据
# ============================================================

with INPUT_PATH.open("r", encoding="utf-8") as f:
    original_history = json.load(f)

# 深复制，确保不修改内存中的原始对象
demo_history = copy.deepcopy(original_history)

epochs = np.asarray(
    [item["epoch"] for item in original_history],
    dtype=float,
)

original_recall = np.asarray(
    [
        item["validation"]["sample_recall"]
        for item in original_history
    ],
    dtype=float,
)


# ============================================================
# 从原曲线提取波动形态
# ============================================================

smooth_original = moving_average(
    original_recall,
    window=7,
)

residual = original_recall - smooth_original

# 标准化原曲线残差
residual_std = np.std(residual)

if residual_std > 1e-12:
    residual = residual / residual_std
else:
    residual = np.zeros_like(residual)

# 限制极端波动
residual = np.clip(
    residual,
    -1.8,
    1.8,
)

rng = np.random.default_rng(RANDOM_SEED)

# 添加很小的确定性随机扰动，使曲线自然一些
small_noise = rng.normal(
    loc=0.0,
    scale=0.0012,
    size=len(epochs),
)


# ============================================================
# 构造仅用于组会示意的Recall趋势
# ============================================================

# 指数收敛趋势：
# epoch 0约为START_RECALL，后期趋近TARGET_RECALL
trend = (
    START_RECALL
    + (TARGET_RECALL - START_RECALL)
    * (1.0 - np.exp(-epochs / TAU))
)

# 前期波动略大，后期波动较小
fluctuation_scale = (
    0.0035
    + 0.0025 * np.exp(-epochs / 20.0)
)

demo_recall = (
    trend
    + residual * fluctuation_scale
    + small_noise
)

# 限制到合理范围
demo_recall = np.clip(
    demo_recall,
    0.70,
    0.875,
)

# 让前几轮整体呈现上升趋势，避免出现突兀下降
for i in range(1, min(8, len(demo_recall))):
    lower_bound = demo_recall[i - 1] - 0.003
    demo_recall[i] = max(
        demo_recall[i],
        lower_bound,
    )

# 按定义同步计算漏选率
demo_miss_rate = 1.0 - demo_recall


# ============================================================
# 写入Demo副本
# ============================================================

for item, new_recall, new_mr in zip(
    demo_history,
    demo_recall,
    demo_miss_rate,
):
    validation = item["validation"]

    # 保存原始真实数据，便于随时恢复和核查
    validation["sample_recall_original"] = (
        validation["sample_recall"]
    )

    validation["miss_selection_rate_original"] = (
        validation["miss_selection_rate"]
    )

    # 写入临时示意值
    validation["sample_recall"] = float(new_recall)
    validation["miss_selection_rate"] = float(new_mr)

    # 加入醒目标记
    item["_demo_only"] = True
    item["_demo_note"] = (
        "临时组会示意数据，不是实际实验结果，"
        "不得用于论文最终结果或正式投稿。"
    )


with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(
        demo_history,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 生成配套说明
# ============================================================

notice = f"""临时示意文件说明

文件：{OUTPUT_PATH.name}

本文件仅用于组会展示预期曲线形态，不是实际实验结果。
原始实验记录保存在：{INPUT_PATH.name}

临时示意设置：
- 后期Sample Recall目标约为：{TARGET_RECALL:.3f}
- 后期漏选率目标约为：{1.0 - TARGET_RECALL:.3f}
- Recall与漏选率满足：MR = 1 - Sample Recall

组会结束后必须：
1. 重新运行正式实验；
2. 使用真实逐轮结果替换本文件；
3. 删除论文中的“示意曲线”；
4. 核对history、validation和test结果的一致性。
"""

NOTICE_PATH.write_text(
    notice,
    encoding="utf-8",
)


# ============================================================
# 输出检查结果
# ============================================================

best_epoch = 95

print("=" * 60)
print("临时示意文件已生成")
print("=" * 60)

print(f"原始文件：{INPUT_PATH.resolve()}")
print(f"示意文件：{OUTPUT_PATH.resolve()}")
print(f"说明文件：{NOTICE_PATH.resolve()}")

print("\n后10轮示意结果：")

for epoch, recall, mr in zip(
    epochs[-10:].astype(int),
    demo_recall[-10:],
    demo_miss_rate[-10:],
):
    print(
        f"Epoch {epoch:02d}: "
        f"Recall={recall:.4f}, "
        f"MR={mr:.4f}"
    )

if best_epoch < len(demo_history):
    best_val = demo_history[best_epoch]["validation"]

    print(f"\nEpoch {best_epoch}：")
    print(
        "原始Recall："
        f"{best_val['sample_recall_original']:.4f}"
    )
    print(
        "示意Recall："
        f"{best_val['sample_recall']:.4f}"
    )
    print(
        "原始MR："
        f"{best_val['miss_selection_rate_original']:.4f}"
    )
    print(
        "示意MR："
        f"{best_val['miss_selection_rate']:.4f}"
    )

print("\n注意：该文件仅可作为明确标注的临时示意图数据。")
