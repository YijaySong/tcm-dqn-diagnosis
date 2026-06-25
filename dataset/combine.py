#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将“病例1.docx”“病例2.docx”“病例3.docx”整合为
“广安门冠心病辨证数据集.txt”和
“广安门冠心病辨证数据集（不含主述）.txt”。

输出格式：
编号 主述 刻下症 证候要素 证候
1 xxx xxx xxx xxx

不含主述输出格式：
编号 刻下症 证候要素 证候
1 xxx xxx xxx

标点规范：
1. 每行共 5 列，列与列之间只用一个空格分隔；
2. 每列内部的实体词之间统一使用英文逗号 “,” 分隔；
3. 删除每列末尾的句号、逗号等标点；
4. 删除中文逗号、顿号、句号、分号等其他标点符号。
"""

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import re


# Word 文档 XML 命名空间
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# 输入和输出文件名
SOURCE_DOCS = ["病例1.docx", "病例2.docx", "病例3.docx"]
OUTPUT_FILE = "广安门冠心病辨证数据集.txt"
OUTPUT_FILE_WITHOUT_CHIEF_COMPLAINT = "广安门冠心病辨证数据集（不含主述）.txt"

# 当前脚本所在目录，即 dataset 目录。
# 若在交互环境中执行本文件内容，__file__ 可能不是 combine.py，
# 此时自动回退到当前工作目录下的 dataset 目录。
DATASET_DIR = Path(__file__).resolve().parent
if not all((DATASET_DIR / doc_name).exists() for doc_name in SOURCE_DOCS):
    DATASET_DIR = Path.cwd() / "dataset"

# 需要统一处理的标点：这些符号会被视为实体词分隔符，最终转换为英文逗号
PUNCTUATION_PATTERN = re.compile(r"[，,。．\.、；;：:！!？?（）()【】\[\]《》<>“”\"‘’'、/\\|·…—\-~～]+")


# 从 docx 文件中读取所有非空段落文本
# docx 本质上是 zip 压缩包，正文保存在 word/document.xml 中
# 使用标准库解析，避免依赖 python-docx 等第三方库

def extract_paragraphs(docx_path: Path) -> list[str]:
    with ZipFile(docx_path) as docx_zip:
        document_xml = docx_zip.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:p", WORD_NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)).strip()
        if text:
            paragraphs.append(text)

    return paragraphs


# 清理字段内容：
# 1. 去掉所有空白字符，防止字段内部出现空格导致列错位；
# 2. 将所有标点统一替换为英文逗号；
# 3. 合并连续英文逗号；
# 4. 删除字段开头和末尾的英文逗号，确保每列末尾没有标点。

def normalize_field(text: str) -> str:
    text = re.sub(r"\s+", "", text.strip())
    text = PUNCTUATION_PATTERN.sub(",", text)
    text = re.sub(r",+", ",", text)
    return text.strip(",")


# 从一个 Word 病例文件中解析病例数据。
# 每条病例通常包含：编号、主诉、刻下症、辨证。
# 部分 docx 中一条病例可能被放在同一个段落内，因此这里同时兼容：
# - 单独编号段落，如 “001”；
# - 编号与主诉连在一起，如 “001主诉：...”

def parse_docx_cases(docx_path: Path) -> list[tuple[str, str, str, str]]:
    paragraphs = extract_paragraphs(docx_path)

    # 找出每条病例的起始段落位置
    starts = [
        index
        for index, paragraph in enumerate(paragraphs)
        if re.fullmatch(r"\d{3}", paragraph) or re.match(r"^\d{3}\s*主诉[：:]", paragraph)
    ]

    cases: list[tuple[str, str, str, str]] = []
    errors: list[str] = []

    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(paragraphs)

        # 将一条病例范围内的段落合并，便于统一正则提取
        block = " ".join(paragraphs[start:end])

        # 去掉病例原始三位编号，最终输出会重新生成连续编号
        id_match = re.match(r"^(\d{3})\s*", block)
        rest = block[id_match.end():] if id_match else block

        # 个别源文件存在不完整的重复片段；若同一块中出现多个“主诉：”，保留最后一个完整病例
        last_main_index = max(rest.rfind("主诉："), rest.rfind("主诉:"))
        if last_main_index > 0:
            rest = rest[last_main_index:]

        # 提取主诉、刻下症、证候要素、证候四个字段
        case_pattern = re.compile(
            r"主诉[：:]\s*(.*?)\s*"
            r"刻下症[：:]\s*(.*?)\s*"
            r"辨证[：:]\s*证候要素[：:]\s*(.*?)\s+"
            r"证候[：:]\s*(.*?)\s*$"
        )
        match = case_pattern.search(rest)

        if not match:
            errors.append(block[:300])
            continue

        chief_complaint, current_symptoms, syndrome_elements, syndrome = [
            normalize_field(field) for field in match.groups()
        ]
        cases.append((chief_complaint, current_symptoms, syndrome_elements, syndrome))

    if errors:
        raise RuntimeError(f"{docx_path.name} 解析失败 {len(errors)} 条，第一条异常内容：{errors[0]}")

    return cases


# 汇总三个病例文件，并写入目标 txt 数据集

def build_dataset() -> None:
    all_cases: list[tuple[str, str, str, str]] = []

    # 依次读取三个 Word 文件，保持原始文件顺序
    for doc_name in SOURCE_DOCS:
        docx_path = DATASET_DIR / doc_name
        all_cases.extend(parse_docx_cases(docx_path))

    # 写入完整数据集表头；列与列之间只用空格分隔，不使用标点分隔列
    lines = ["编号 主述 刻下症 证候要素 证候"]

    # 写入不含主述数据集表头，只保留编号、刻下症、证候要素、证候四列
    lines_without_chief_complaint = ["编号 刻下症 证候要素 证候"]

    # 重新生成从 1 开始的连续编号，并同步构造两个数据集
    for index, (chief_complaint, current_symptoms, syndrome_elements, syndrome) in enumerate(all_cases, start=1):
        full_row = [str(index), chief_complaint, current_symptoms, syndrome_elements, syndrome]
        row_without_chief_complaint = [str(index), current_symptoms, syndrome_elements, syndrome]
        lines.append(" ".join(full_row))
        lines_without_chief_complaint.append(" ".join(row_without_chief_complaint))

    output_path = DATASET_DIR / OUTPUT_FILE
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_path_without_chief_complaint = DATASET_DIR / OUTPUT_FILE_WITHOUT_CHIEF_COMPLAINT
    output_path_without_chief_complaint.write_text("\n".join(lines_without_chief_complaint) + "\n", encoding="utf-8")

    print(f"已生成：{output_path}")
    print(f"已生成：{output_path_without_chief_complaint}")
    print(f"病例数：{len(all_cases)}")


if __name__ == "__main__":
    build_dataset()
