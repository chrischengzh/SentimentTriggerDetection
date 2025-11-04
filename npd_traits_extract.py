# Python
"""
单句 NPD 特征检测（词典+零样本+配偶锚点过滤）
- 输入一条文本
- 如命中“配偶/伴侣”锚点才进行检测
- 词典规则打分 + 零样本打分 → 融合 → 打印结果
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
PARTNER_HINTS_PATH = os.path.join(DATA_DIR, "partner_hints.json")

# 由 partner_hints.json 构建配偶/伴侣锚点正则
def _load_partner_hints(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):which brew
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
    model_id = "valhalla/distilbart-mnli-12-1"
    # model_id = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI"
    # model_id = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    # model_id = "joeddav/xlm-roberta-large-xnli"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)  # 避免 tiktoken 转换
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline(
        "zero-shot-classification",
        model=mdl,
        tokenizer=tok,
        device=device_id,
        tokenizer_kwargs={"truncation": True, "max_length": 512}
    )

# 从词典文件构建零样本候选标签与映射
def build_zero_shot_labels_from_lexicon(json_path: str):
    # data = load_trait_lexicon(json_path)
    # 需要原始 JSON 以获取 label/desc 文本，因此重新读取原文件
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = []
    desc2key = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        # 优先使用中文描述 desc.zh；若无，则退化到 label.zh；再退到英文 desc.en/label.en
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

def zero_shot_scores(zs, text: str) -> dict: # 接收一个已构建好的零样本分类 pipeline 实例 zs，以及待分析文本 text
    if not text or len(text) < 8: # 输入校验：如果文本为空或长度小于 8，直接返回空字典，避免无意义/噪声评分。
        return {}
    out = zs(text, ZERO_SHOT_LABELS, multi_label=True) # 使用一组候选标签 ZERO_SHOT_LABELS 做多标签分类（每个标签独立给置信度）
    scores = {}
    for label, score in zip(out["labels"], out["scores"]): # 结果整理：out["labels"] 是与候选标签对应的描述文本列表，out["scores"] 是各自的概率/置信度
        k = DESC2KEY[label]
        scores[k] = float(score)
    return scores

# ---------- 分数融合 ----------
def fuse_scores(lex_scores: dict, zs_scores: dict, w_lex=0.4, w_zs=0.6) -> dict: # lex_scores：词典/规则打分，zs_scores：零样本模型打分 dict，w_lex、w_zs：各自权重。
    keys = set(lex_scores) | set(zs_scores)
    fused = {}
    for k in keys:
        fused[k] = w_lex * lex_scores.get(k, 0.0) + w_zs * zs_scores.get(k, 0.0)
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

    # 锚点过滤
    # if not has_partner_anchor(text):
    #     print("未检测到配偶/伴侣锚点，跳过以降低误报。")
    #     return

    # 词典与正则
    if not os.path.exists(TRAIT_LEXICON_PATH):
        print(f"找不到词典文件: {TRAIT_LEXICON_PATH}")
        sys.exit(1)
    lexicon = load_trait_lexicon(TRAIT_LEXICON_PATH)
    re_traits = build_regexes(lexicon)

    # 词典打分
    lex_scores = lexicon_scores(text, re_traits)
    # 将英文 trait 键映射为中文标签用于展示
    with open(TRAIT_LEXICON_PATH, "r", encoding="utf-8") as _f:
        _raw_lex = json.load(_f)
    KEY2ZH = {k: (v.get("label") or {}).get("zh", k) for k, v in _raw_lex.items() if isinstance(v, dict)}
    lex_scores_cn = {KEY2ZH.get(k, k): v for k, v in lex_scores.items()}

    # Zero-shot
    # 自动设备选择（GPU优先，CPU 为 -1）
    device_id = 0 if os.getenv("HF_DEVICE", "auto") != "cpu" else -1
    try:
        zs = build_zero_shot(device_id)
    except Exception:
        # 兜底到 CPU
        zs = build_zero_shot(-1)

    # 使用词典文件构建候选标签与映射，替代硬编码的 LABEL_TO_DESC
    global ZERO_SHOT_LABELS, DESC2KEY
    ZERO_SHOT_LABELS, DESC2KEY = build_zero_shot_labels_from_lexicon(TRAIT_LEXICON_PATH)

    zs_scores = zero_shot_scores(zs, text[:1000])

    # 融合与阈值
    fused = fuse_scores(lex_scores, zs_scores, w_lex=0.4, w_zs=0.6)
    picks = {k: v for k, v in fused.items() if v >= 0.2}

    # 打印结果
    def sort_items(d): return sorted(((k, round(v, 3)) for k, v in d.items()), key=lambda x: -x[1])

    print("\n输入：", text)
    print("\n词典匹配得分（中文标签）:")
    print(sort_items(lex_scores_cn) or "无命中")

    print("\nZero-shot top5:")
    print(sort_items(zs_scores)[:5] or "无命中")

    print("\nFused traits (>=0.2):") # 可以根据需要调整识别的阈值
    print(sort_items(picks) or "无")

if __name__ == "__main__":
    main()