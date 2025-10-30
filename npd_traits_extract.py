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
from transformers import pipeline

# ---------- 配置 ----------
DATA_DIR = "data"
TRAIT_LEXICON_PATH = os.path.join(DATA_DIR, "trait_lexicon.json")

# 配偶/伴侣锚点（中英混合）
PARTNER_HINTS = [
    r"\b(my|our)\s+(husband|wife|spouse|partner|bf|boyfriend|gf|girlfriend)\b",
    r"\b(ex[- ]?(husband|wife|partner|bf|boyfriend|gf|girlfriend))\b",
    r"(我(的)?|我们(的)?|咱们(的)?|我家|我们家|咱家)\s*(丈夫|老公|先生|老伴|妻子|老婆|太太|贤内助|男女朋友|男友|女友|男朋友|女朋友|对象|伴侣|配偶|爱人|另一半)",
    r"(前任|前夫|前妻|前男友|前女友|前男朋友|前女朋友|前对象|前伴侣|前配偶)",
    r"(他|她|他的|她的|ta|TA)"
]
re_PARTNER = [re.compile(p, re.I) for p in PARTNER_HINTS]

def has_partner_anchor(text: str) -> bool:
    return any(rp.search(text) for rp in re_PARTNER)

# ---------- 词典加载与编译（对中文移除 \b） ----------
def load_trait_lexicon(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("trait_lexicon.json 必须是 {trait: [regex,...]} 结构")
    return data

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

# ---------- Zero-shot 多标签 ----------
LABEL_TO_DESC = {
    "grandiosity": "shows grandiosity or superiority",
    "自大夸大": "表现出自大或优越感",
    "fantasy_of_success_power_beauty": "has fantasies of unlimited success, power, beauty or ideal love",
    "成功权力美貌幻想": "沉溺于无限成功、权力、美貌或理想爱情的幻想",
    "special_unique": "believes they are special and unique",
    "特殊独一无二": "相信自己特殊而独一无二",
    "need_for_admiration": "needs excessive admiration",
    "需要崇拜赞美": "需要过度的崇拜与赞美",
    "entitlement": "has a sense of entitlement",
    "特权感理应拥有": "具有特权感，认为理应拥有一切",
    "exploitative": "is interpersonally exploitative",
    "利用剥削": "在人际关系中具有剥削与利用倾向",
    "lack_of_empathy": "lacks empathy",
    "缺乏同理心": "缺乏同理心",
    "envious_arrogant": "is envious or arrogant",
    "嫉妒傲慢": "表现出嫉妒或傲慢",
    "love_bombing": "uses love bombing",
    "爱情轰炸": "使用爱情轰炸策略",
    "gaslighting": "gaslights others",
    "煤气灯操控": "对他人实施煤气灯操控",
    "triangulation": "uses triangulation",
    "三角关系操纵": "通过引入第三者进行三角操纵",
    "devaluation_discard": "devalues or discards partners",
    "贬低抛弃": "对伴侣进行贬低或抛弃",
    "silent_treatment": "uses silent treatment",
    "冷暴力": "以冷处理/冷暴力对待他人",
    "blame_shifting": "engages in blame shifting",
    "推卸责任": "推卸责任、甩锅他人",
    "future_faking": "does future faking",
    "画饼式承诺": "用画饼式承诺进行欺骗",
    "hoovering": "does hoovering",
    "回吸拽回": "通过回吸把人再次拽回关系中"
}
ZERO_SHOT_LABELS = list(LABEL_TO_DESC.values())
DESC2KEY = {v: k for k, v in LABEL_TO_DESC.items()}

def build_zero_shot(device_id: int = 0):
    return pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=device_id,
        tokenizer_kwargs={"truncation": True, "max_length": 384}
    )

def zero_shot_scores(zs, text: str) -> dict:
    if not text or len(text) < 8:
        return {}
    out = zs(text, ZERO_SHOT_LABELS, multi_label=True)
    scores = {}
    for label, score in zip(out["labels"], out["scores"]):
        k = DESC2KEY[label]
        scores[k] = float(score)
    return scores

# ---------- 分数融合 ----------
def fuse_scores(lex_scores: dict, zs_scores: dict, w_lex=0.4, w_zs=0.6) -> dict:
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
    if not has_partner_anchor(text):
        print("未检测到配偶/伴侣锚点，跳过以降低误报。")
        return

    # 词典与正则
    if not os.path.exists(TRAIT_LEXICON_PATH):
        print(f"找不到词典文件: {TRAIT_LEXICON_PATH}")
        sys.exit(1)
    lexicon = load_trait_lexicon(TRAIT_LEXICON_PATH)
    re_traits = build_regexes(lexicon)

    # 词典打分
    lex_scores = lexicon_scores(text, re_traits)

    # Zero-shot
    # 自动设备选择（GPU优先，CPU 为 -1）
    device_id = 0 if os.getenv("HF_DEVICE", "auto") != "cpu" else -1
    try:
        zs = build_zero_shot(device_id)
    except Exception:
        # 兜底到 CPU
        zs = build_zero_shot(-1)
    zs_scores = zero_shot_scores(zs, text[:1000])

    # 融合与阈值
    fused = fuse_scores(lex_scores, zs_scores, w_lex=0.4, w_zs=0.6)
    picks = {k: v for k, v in fused.items() if v >= 0.45}

    # 打印结果
    def sort_items(d): return sorted(((k, round(v, 3)) for k, v in d.items()), key=lambda x: -x[1])

    print("\n输入：", text)
    print("\nLexicon scores:")
    print(sort_items(lex_scores) or "无命中")

    print("\nZero-shot top5:")
    print(sort_items(zs_scores)[:5] or "无命中")

    print("\nFused traits (>=0.45):")
    print(sort_items(picks) or "无")

if __name__ == "__main__":
    main()