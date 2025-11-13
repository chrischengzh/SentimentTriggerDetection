#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_npd_traits_statistics.py
--------------------------------
Reads reddit_posts_comments_npd.csv and counts:
1) Personal traits in column 'traits'
2) Manipulative behaviors in column 'manipulations'

Both columns may contain:
- Python literal lists: ['a', 'b']
- JSON lists: ["a", "b"]
- String lists: "a, b, c"

Outputs counts sorted from most to least.
"""

import pandas as pd
import ast
import json
from pathlib import Path
import argparse


def parse_list_cell(cell):
    """
    Parse a cell that may contain:
    - Python literal list
    - JSON list
    - Comma-separated string
    Returns: a list of strings
    """
    if cell is None:
        return []

    if isinstance(cell, list):
        return [str(x).strip() for x in cell]

    s = str(cell).strip()
    if not s:
        return []

    # Try Python literal eval
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [str(x).strip() for x in obj]
    except Exception:
        pass

    # Try JSON parse
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj]
    except Exception:
        pass

    # Fallback: comma split
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]

    # Single entry
    return [s]


def count_items(series):
    """
    Count frequency of all items in a pandas Series of list-like strings.
    """
    counter = {}
    for cell in series.dropna().tolist():
        items = parse_list_cell(cell)
        for item in items:
            counter[item] = counter.get(item, 0) + 1

    # Sort by count descending
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Count traits & manipulations from CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="training/reddit_posts_comments_npd.csv",
        help="CSV path (default: training/reddit_posts_comments_npd.csv)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[Error] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Check required columns
    if "traits" not in df.columns:
        print("[Error] CSV missing 'traits' column.")
        return
    if "manipulations" not in df.columns:
        print("[Error] CSV missing 'manipulations' column.")
        return

    print("\n===== Personal Traits Statistics (traits column) =====")
    traits_counts = count_items(df["traits"])
    for item, cnt in traits_counts:
        print(f"{item}: {cnt}")

    print("\n===== Relationship Manipulation Statistics (manipulations column) =====")
    manip_counts = count_items(df["manipulations"])
    for item, cnt in manip_counts:
        print(f"{item}: {cnt}")


if __name__ == "__main__":
    main()