#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 lhz_data.xlsx 生成 lhz_data.txt。

输出格式：
id A列内容 B列内容
1 xxx xxx

处理规则：
1. 输出共 3 列，列与列之间只用一个空格分隔；
2. 第一列为从 1 开始的连续 id，第二列来自 Excel A 列，第三列来自 Excel B 列；
3. 如果 B 列为空且同一行 H 列有内容，则用 H 列内容填充第三列；
4. 如果 B 列与同一行 H 列不同，则把 H 列中有而 B 列没有的词补充到第三列，用英文逗号分隔；
5. 每列内部词语统一用英文逗号分隔，并删除每列末尾标点。
"""

from __future__ import annotations

from pathlib import Path
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET


INPUT_FILE = "lhz_data.xlsx"
OUTPUT_FILE = "lhz_data.txt"

DATASET_DIR = Path(__file__).resolve().parent
INPUT_PATH = DATASET_DIR / INPUT_FILE
OUTPUT_PATH = DATASET_DIR / OUTPUT_FILE

# 这些符号会被视为词语分隔符，最终统一转换为英文逗号。
PUNCTUATION_PATTERN = re.compile(r"[，,。．\.、；;：:！!？?（）()【】\[\]《》<>“”\"‘’'、/\\|·…—\-~～]+")
SPACE_PATTERN = re.compile(r"\s+")

# Excel xlsx 内部 XML 命名空间。
SPREADSHEET_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}


# 清理字段：去空白、统一标点为英文逗号、合并连续逗号、删除首尾逗号。
def normalize_field(value: object | None) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    # Excel 中的整数有时会被读成 1.0，这里转回 1。
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    text = SPACE_PATTERN.sub("", text)
    text = PUNCTUATION_PATTERN.sub(",", text)
    text = re.sub(r",+", ",", text)
    return text.strip(",")


# 按英文逗号切分字段，同时保留原顺序并去重。
def split_words(text: str) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()

    for word in normalize_field(text).split(","):
        word = word.strip(",")
        if word and word not in seen:
            words.append(word)
            seen.add(word)

    return words


# 将同一行 H 列中存在、但 B 列中不存在的词补充到第三列。
def merge_b_with_h(column_b: object | None, column_h: object | None) -> str:
    b_words = split_words(normalize_field(column_b))
    h_words = split_words(normalize_field(column_h))

    if not b_words:
        return ",".join(h_words)

    merged_words = b_words[:]
    seen = set(merged_words)
    for word in h_words:
        if word not in seen:
            merged_words.append(word)
            seen.add(word)

    return ",".join(merged_words)


# 读取 xlsx 中的 sharedStrings.xml。
def read_shared_strings(xlsx_zip: ZipFile) -> list[str]:
    try:
        shared_strings_xml = xlsx_zip.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(shared_strings_xml)
    strings: list[str] = []

    for item in root.findall("main:si", SPREADSHEET_NS):
        texts = [node.text or "" for node in item.findall(".//main:t", SPREADSHEET_NS)]
        strings.append("".join(texts))

    return strings


# 找到第一个工作表对应的 XML 文件路径。
def find_first_sheet_path(xlsx_zip: ZipFile) -> str:
    workbook_xml = xlsx_zip.read("xl/workbook.xml")
    workbook_root = ET.fromstring(workbook_xml)
    first_sheet = workbook_root.find("main:sheets/main:sheet", SPREADSHEET_NS)
    if first_sheet is None:
        raise RuntimeError(f"{INPUT_FILE} 中没有工作表")

    relationship_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]

    rels_xml = xlsx_zip.read("xl/_rels/workbook.xml.rels")
    rels_root = ET.fromstring(rels_xml)
    for relationship in rels_root.findall("rel:Relationship", REL_NS):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"]
            if target.startswith("/"):
                return target.lstrip("/")
            return f"xl/{target}"

    raise RuntimeError(f"未找到第一个工作表的关系：{relationship_id}")


# 将 Excel 单元格地址（如 A1、B12、H3）转换为列名（A、B、H）。
def get_column_name(cell_reference: str) -> str:
    return re.sub(r"\d+", "", cell_reference)


# 读取单元格真实文本。
def read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")

    # inlineStr 类型直接保存在单元格内部。
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//main:t", SPREADSHEET_NS)]
        return "".join(texts)

    value_node = cell.find("main:v", SPREADSHEET_NS)
    if value_node is None or value_node.text is None:
        return ""

    raw_value = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw_value)]

    return raw_value


# 用标准库读取 xlsx，避免依赖 openpyxl。
def read_xlsx_rows(xlsx_path: Path) -> list[dict[str, str]]:
    with ZipFile(xlsx_path) as xlsx_zip:
        shared_strings = read_shared_strings(xlsx_zip)
        sheet_path = find_first_sheet_path(xlsx_zip)
        sheet_xml = xlsx_zip.read(sheet_path)

    sheet_root = ET.fromstring(sheet_xml)
    rows: list[dict[str, str]] = []

    for row in sheet_root.findall(".//main:sheetData/main:row", SPREADSHEET_NS):
        row_values: dict[str, str] = {}
        for cell in row.findall("main:c", SPREADSHEET_NS):
            cell_reference = cell.attrib.get("r", "")
            column_name = get_column_name(cell_reference)
            if column_name in {"A", "B", "H"}:
                row_values[column_name] = read_cell_value(cell, shared_strings)
        rows.append(row_values)

    return rows


# 判断是否为表头行：如果首行是常见列名，则不作为病例数据写出。
def is_header_row(row: dict[str, str]) -> bool:
    normalized_values = {normalize_field(value).lower() for value in row.values() if normalize_field(value)}
    header_words = {"id", "编号", "症状", "刻下症", "证候", "证候要素", "a", "b", "h"}
    return bool(normalized_values & header_words)


# 生成 lhz_data.txt。
def build_lhz_data() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"未找到输入文件：{INPUT_PATH}")

    rows = read_xlsx_rows(INPUT_PATH)
    output_lines = ["id symptom Se"]

    data_index = 1
    skipped_empty_column = 0
    for row_index, row in enumerate(rows, start=1):
        # 跳过第一行表头，以及 A/B/H 三列均为空的行。
        if row_index == 1 and is_header_row(row):
            continue

        symptom = normalize_field(row.get("A"))
        Se = merge_b_with_h(row.get("B"), row.get("H"))

        # 输出数据集要求每行三列都有内容；任一列为空则删除该行。
        if not symptom or not Se:
            skipped_empty_column += 1
            continue

        output_lines.append(" ".join([str(data_index), symptom, Se]))
        data_index += 1

    OUTPUT_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT_PATH}")
    print(f"数据量：{data_index - 1}")
    print(f"跳过空列行数：{skipped_empty_column}")


if __name__ == "__main__":
    build_lhz_data()
