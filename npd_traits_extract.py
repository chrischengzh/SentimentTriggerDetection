# Python
"""
单句 NPD 特征检测（词典+零样本+配偶锚点过滤）
- 输入一条文本
- 如命中“配偶/伴侣”锚点才进行检测
- 词典规则打分 + 零样本打分（特征+关系操控）→ 融合 → 打印结果
依赖：transformers、pandas(无硬依赖，仅为一致性可不装)
"""

import os
import re
import json
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------- 配置 ----------
DATA_DIR = "data"
TRAIT_LEXICON_PATH = os.path.join(DATA_DIR, "npd_trait_lexicons.json")
MANIPULAATION_LEXICON_PATH = os.path.join(DATA_DIR, "npd_manipulation_lexicons.json")
PARTNER_HINTS_PATH = os.path.join(DATA_DIR, "partner_hints.json")

# 由 partner_hints.json 构建配偶/伴侣锚点正则
def _load_partner_hints(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        ph = data.get("partner_hints")
        if isinstance(ph, dict):
            return list(ph.get("zh") or []) + list(ph.get("en") or [])
        # 兼容顶层 zh/en 或 patterns.zh/en
        combined = []
        for key in ("zh", "en"):
            v = data.get(key)
            if isinstance(v, list):
                combined += v
        if combined:
            return combined
        patterns = data.get("patterns") or {}
        combined = list(patterns.get("zh") or []) + list(patterns.get("en") or [])
        if combined:
            return combined
    raise ValueError("partner_hints.json 格式不支持")

try:
    _partner_hints = _load_partner_hints(PARTNER_HINTS_PATH)
except Exception:
    _partner_hints = []
re_PARTNER = [re.compile(p, re.I) for p in _partner_hints if isinstance(p, str) and p.strip()]

def has_partner_anchor(text: str) -> bool:
    return any(rp.search(text) for rp in re_PARTNER)

# ---------- 词典加载与编译（对中文移除 \b） ----------
def load_trait_lexicon(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            "npd_trait_lexicons.json 必须是 {trait: {...}} 字典，且正则放在 顶层 zh 或 patterns.zh 中"
        )

    zh_only = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        patterns_zh = None
        # 1) 兼容老格式：顶层 zh 为正则数组
        if isinstance(v.get("zh"), list):
            patterns_zh = v["zh"]
        # 2) 兼容新格式：patterns.zh 为正则数组
        if patterns_zh is None:
            patterns = v.get("patterns", {})
            if isinstance(patterns.get("zh"), list):
                patterns_zh = patterns["zh"]
        # 3) 兼容另一命名：regex.zh
        if patterns_zh is None:
            regex_block = v.get("regex", {})
            if isinstance(regex_block.get("zh"), list):
                patterns_zh = regex_block["zh"]

        if isinstance(patterns_zh, list) and patterns_zh:
            zh_only[k] = patterns_zh

    if not zh_only:
        raise ValueError(
            "未能从词典中解析到任何可用的中文正则（支持顶层 zh 或 patterns.zh 或 regex.zh）"
        )
    return zh_only

def _fix_cn_pattern(p: str) -> str:
    if any('\u4e00' <= ch <= '\u9fff' for ch in p):
        return p.replace(r"\b", "")
    return p

def build_regexes(lexicon: dict) -> dict:
    return {k: [re.compile(_fix_cn_pattern(p), re.I) for p in v] for k, v in lexicon.items()}

def lexicon_scores(text: str, re_traits: dict) -> dict:
    scores = {}
    for trait, patterns in re_traits.items():
        hit = sum(1 for rp in patterns if rp.search(text))
        if hit > 0:
            scores[trait] = min(1.0, 0.2 * hit)
    return scores

def build_zero_shot(device_id: int = -1):
    # model_id = "uer/sbert-base-chinese-nli"
    # model_id = "valhalla/distilbart-mnli-12-1"
    # model_id = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI"
    # model_id = "joeddav/xlm-roberta-large-xnli"
    model_id = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)  # 避免 tiktoken 转换
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline(
        "zero-shot-classification",
        model=mdl,
        tokenizer=tok,
        device=device_id,
        tokenizer_kwargs={"truncation": True, "max_length": 512}
    )

# 从词典文件构建零样本候选标签与映射（特征）
def build_zero_shot_labels_from_lexicon(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = []
    desc2key = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        desc_zh = (v.get("desc") or {}).get("zh")
        label_zh = (v.get("label") or {}).get("zh")
        desc_en = (v.get("desc") or {}).get("en")
        label_en = (v.get("label") or {}).get("en")
        chosen = desc_zh or label_zh or desc_en or label_en
        if not chosen:
            continue
        labels.append(chosen)
        desc2key[chosen] = (v.get("label") or {}).get("zh") or k
    if not labels:
        raise ValueError("未能从 npd_trait_lexicons.json 中提取到任何候选标签/描述")
    return labels, desc2key

# 从操控词典文件构建零样本候选标签与映射（关系操控）
def build_zero_shot_labels_from_manip_lexicon(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = []
    desc2key = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        desc_zh = (v.get("desc") or {}).get("zh")
        label_zh = (v.get("label") or {}).get("zh")
        desc_en = (v.get("desc") or {}).get("en")
        label_en = (v.get("label") or {}).get("en")
        chosen = desc_zh or label_zh or desc_en or label_en
        if not chosen:
            continue
        labels.append(chosen)
        desc2key[chosen] = (v.get("label") or {}).get("zh") or k
    if not labels:
        raise ValueError("未能从 npd_manipulation_lexicons.json 中提取到任何候选标签/描述")
    return labels, desc2key

# 构建 英文键 -> 中文显示名 的映射
def build_key2zh_map(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    key2zh = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            key2zh[k] = (v.get("label") or {}).get("zh", k)
    return key2zh

def zero_shot_scores(zs, text: str) -> dict:
    # 特征：使用 ZERO_SHOT_LABELS（traits）
    if not text or len(text) < 8:
        return {}
    out = zs(text, ZERO_SHOT_LABELS, multi_label=True)
    scores = {}
    for label, score in zip(out["labels"], out["scores"]):
        k = DESC2KEY[label]
        scores[k] = float(score)
    return scores

def zero_shot_scores_manip(zs, text: str) -> dict:
    # 关系操控：使用 ZERO_SHOT_LABELS_MANIP
    if not text or len(text) < 8:
        return {}
    out = zs(text, ZERO_SHOT_LABELS_MANIP, multi_label=True)
    scores = {}
    for label, score in zip(out["labels"], out["scores"]):
        k = DESC2KEY_MANIP[label]
        scores[k] = float(score)
    return scores

# ---------- 分数融合 ----------
def fuse_scores(lex_scores: dict, zs_scores_traits: dict, zs_scores_manip: dict, w_lex=0.35, w_zs_traits=0.4, w_zs_manip=0.25) -> dict:
    """
    将三路分数融合：
    - 词典（特征）lex_scores
    - 零样本（特征）zs_scores_traits
    - 零样本（关系操控）zs_scores_manip
    """
    keys = set(lex_scores) | set(zs_scores_traits) | set(zs_scores_manip)
    fused = {}
    for k in keys:
        fused[k] = (
            w_lex * lex_scores.get(k, 0.0)
            + w_zs_traits * zs_scores_traits.get(k, 0.0)
            + w_zs_manip * zs_scores_manip.get(k, 0.0)
        )
    return fused

def main():
    # 读取输入
    try:
        text = input("请输入一句待检测的语句：").strip()
    except EOFError:
        text = ""
    if not text:
        print("空输入，退出。")
        return

    # 锚点过滤（如需生效，取消注释）
    # if not has_partner_anchor(text):
    #     print("未检测到配偶/伴侣锚点，跳过以降低误报。")
    #     return

    # 词典与正则（特征）
    if not os.path.exists(TRAIT_LEXICON_PATH):
        print(f"找不到词典文件: {TRAIT_LEXICON_PATH}")
        sys.exit(1)
    lexicon_traits = load_trait_lexicon(TRAIT_LEXICON_PATH)
    re_traits = build_regexes(lexicon_traits)

    # 词典打分（特征）
    lex_scores = lexicon_scores(text, re_traits)

    # 显示名映射（特征、操控）
    KEY2ZH_TRAIT = build_key2zh_map(TRAIT_LEXICON_PATH)
    KEY2ZH_MANIP = build_key2zh_map(MANIPULAATION_LEXICON_PATH)

    # 将词典分数映射为中文显示
    lex_scores_cn = {KEY2ZH_TRAIT.get(k, k): v for k, v in lex_scores.items()}

    # Zero-shot
    device_id = 0 if os.getenv("HF_DEVICE", "auto") != "cpu" else -1
    try:
        zs = build_zero_shot(device_id)
    except Exception:
        zs = build_zero_shot(-1)

    # 构建零样本候选（特征）
    global ZERO_SHOT_LABELS, DESC2KEY
    ZERO_SHOT_LABELS, DESC2KEY = build_zero_shot_labels_from_lexicon(TRAIT_LEXICON_PATH)
    zs_scores_traits = zero_shot_scores(zs, text[:1000])

    # 构建零样本候选（关系操控）
    if not os.path.exists(MANIPULAATION_LEXICON_PATH):
        print(f"找不到操控词典文件: {MANIPULAATION_LEXICON_PATH}")
        sys.exit(1)
    global ZERO_SHOT_LABELS_MANIP, DESC2KEY_MANIP
    ZERO_SHOT_LABELS_MANIP, DESC2KEY_MANIP = build_zero_shot_labels_from_manip_lexicon(MANIPULAATION_LEXICON_PATH)
    zs_scores_manip = zero_shot_scores_manip(zs, text[:1000])

    # 融合与阈值（统一考虑）
    fused = fuse_scores(lex_scores, zs_scores_traits, zs_scores_manip, w_lex=0.35, w_zs_traits=0.4, w_zs_manip=0.25)
    picks = {k: v for k, v in fused.items() if v >= 0.2}

    # 打印结果
    def sort_items(d): return sorted(((k, round(v, 3)) for k, v in d.items()), key=lambda x: -x[1])

    print("\n输入：", text)

    print("\n词典匹配得分（中文标签，特征）:")
    print(sort_items(lex_scores_cn) or "无命中")

    print("\nZero-shot top 3（人格特征）:")
    print(sort_items(zs_scores_traits)[:3] or "无命中")

    print("\nZero-shot top 3（关系操控）:")
    print(sort_items(zs_scores_manip)[:3] or "无命中")

    # 最终融合结果中文显示（优先显示特征词典中的中文名，其次操控词典）
    def to_cn_key(k: str) -> str:
        return KEY2ZH_TRAIT.get(k) or KEY2ZH_MANIP.get(k) or k

    picks_cn = {to_cn_key(k): v for k, v in picks.items()}
    print("\nFused (traits + manipulation, >=0.2):")
    print(sort_items(picks_cn) or "无")

if __name__ == "__main__":
    main()