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
TRAIT_LEXICON_PATH = os.path.join(DATA_DIR, "partner_hints.json")

# re_PARTNER = [re.compile(p, re.I) for p in PARTNER_HINTS]
# def has_partner_anchor(text: str) -> bool:
#     return any(rp.search(text) for rp in re_PARTNER)

# ---------- 词典加载与编译（对中文移除 \b） ----------
def load_trait_lexicon(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            "npd_trait_lexicon.json 必须是 {trait: {...}} 字典，且正则放在 顶层 zh 或 patterns.zh 中"
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

# ---------- Zero-shot 多标签 ----------
LABEL_TO_DESC = {
    "自大夸大": "表现出自大或优越感",
    "幻想成功权力美貌": "沉溺于无限成功、权力、美貌或理想爱情的幻想",
    "特殊且独一无二": "相信自己特殊而独一无二",
    "需要崇拜赞美": "需要过度的崇拜与赞美",
    "特权感且理所当然": "具有特权感，认为理应拥有一切",
    "利用和剥削他人": "在人际关系中具有剥削与利用倾向",
    "缺乏同理心": "缺乏同理心，不关心他人的感受",
    "嫉妒傲慢": "表现出嫉妒或傲慢",

    "精神操控": "对他人实施精神操控",
    "狂热示好": "使用爱情、关心等轰炸策略",
    "情感虐待": "在情感上虐待伴侣",
    "挑拨离间": "通过引入第三者进行三角关系操纵",
    "否定贬低羞辱": "对伴侣进行贬低和羞辱，否定他人的价值，否认他人的努力",
    "抛弃": "对伴侣进行抛弃",
    "冷暴力": "以冷处理/冷暴力对待他人",
    "推卸责任": "推卸责任、甩锅他人",
    "画饼式承诺": "用画饼式承诺进行欺骗",
    "回吸拽回": "通过回吸把人再次拽回关系中"
}
ZERO_SHOT_LABELS = list(LABEL_TO_DESC.values())
DESC2KEY = {v: k for k, v in LABEL_TO_DESC.items()}

def build_zero_shot(device_id: int = 0):
    return pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=device_id,
        tokenizer_kwargs={"truncation": True, "max_length": 1024}
    )

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
    picks = {k: v for k, v in fused.items() if v >= 0.2}

    # 打印结果
    def sort_items(d): return sorted(((k, round(v, 3)) for k, v in d.items()), key=lambda x: -x[1])

    print("\n输入：", text)
    print("\nLexicon scores:")
    print(sort_items(lex_scores) or "无命中")

    print("\nZero-shot top5:")
    print(sort_items(zs_scores)[:5] or "无命中")

    print("\nFused traits (>=0.2):") # 可以根据需要调整识别的阈值
    print(sort_items(picks) or "无")

if __name__ == "__main__":
    main()