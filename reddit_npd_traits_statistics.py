#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_npd_traits_statistics.py
--------------------------------
根据以下配置自动定位 CSV 文件：

    training/reddit_posts_comments_{SUBREDDIT}_{POST_LIMIT}.csv

对 CSV 的 traits / manipulations 列进行合并统计：
- total_prob (累计概率)
- count      (出现次数)
- avg_prob   (平均概率)

按 total_prob DESC 排序打印。
"""

import pandas as pd
import ast
import json
from pathlib import Path
from typing import Any, List, Tuple, Optional


# ---------- 配置 ----------
SUBREDDIT = "NPD"       # "Reddit NPD"
POST_LIMIT = 10         # 抓取帖子数
TRAIN_DIR = "training"


def build_csv_path() -> Path:
    """构建与 extract 脚本同步的 CSV 文件名。"""
    filename = f"reddit_posts_comments_{SUBREDDIT}_{POST_LIMIT}.csv"
    return Path(TRAIN_DIR) / filename


def parse_pairs_cell(cell: Any) -> List[Tuple[str, Optional[float]]]:
    """将一个单元格解析为 (label, prob) 列表。"""
    if cell is None:
        return []

    # 已经是 list/tuple（python literal 已 eval）
    if isinstance(cell, (list, tuple)):
        out = []
        for item in cell:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                label = str(item[0]).strip()
                try:
                    prob = float(item[1])
                except:
                    prob = None
                out.append((label, prob))
            elif isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("trait")
                score = item.get("score") or item.get("prob") or item.get("value") or item.get("confidence")
                if label is not None:
                    try:
                        prob = float(score)
                    except:
                        prob = None
                    out.append((label.strip(), prob))
        return out

    s = str(cell).strip()
    if not s:
        return []

    # Python literal list
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return parse_pairs_cell(obj)
    except:
        pass

    # JSON list
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)):
            return parse_pairs_cell(obj)
    except:
        pass

    # fallback
    return [(s, None)]


def aggregate(series: pd.Series):
    """聚合 (label, prob)，按 total_prob 排序。"""
    totals = {}
    counts = {}

    for cell in series.dropna():
        pairs = parse_pairs_cell(cell)
        for label, prob in pairs:
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
            totals[label] = totals.get(label, 0.0) + (float(prob) if prob else 0.0)

    results = []
    for label in counts:
        total = totals[label]
        cnt = counts[label]
        avg = total / cnt if cnt else 0.0
        results.append((label, total, cnt, avg))

    # ⭐ 按 total_prob DESC，再按 count DESC 排序
    results.sort(key=lambda x: (-x[1], -x[2]))
    return results


def main():
    csv_path = build_csv_path()

    if not csv_path.exists():
        print(f"[Error] 文件不存在: {csv_path}")
        return

    print(f"\n[读取统计文件]: {csv_path}")

    df = pd.read_csv(csv_path)

    if "traits" not in df.columns or "manipulations" not in df.columns:
        print("[Error] CSV 必须包含 traits 和 manipulations 列")
        return

    print("\n===== Personal Traits (按累计概率排序) =====")
    traits_stats = aggregate(df["traits"])
    for label, total, cnt, avg in traits_stats:
        print(f"{label}: 概率累计={total:.3f} | 平均={avg:.3f} | 次数={cnt}")

    print("\n===== Manipulation Behaviors (按累计概率排序) =====")
    manip_stats = aggregate(df["manipulations"])
    for label, total, cnt, avg in manip_stats:
        print(f"{label}: 概率累计={total:.3f} | 平均={avg:.3f} | 次数={cnt}")


if __name__ == "__main__":
    main()