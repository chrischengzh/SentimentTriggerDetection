import os
import re
import time
import pandas as pd
import praw
from transformers import pipeline
import json

# ---------- 配置 ----------
SUBREDDIT = "NPD" # "NarcissisticSpouses"
POST_LIMIT = 5  # 抓取帖子数
MAX_COMMENTS = 0    # 评论上限（总量控制）
POST_MAX_LENGTH = 2048 # build_zero_shot的tokenizer的max_length
DATA_DIR = "data"
TRAIN_DIR = "training"
OUT_REDDIT_POSTS_NPD_CSV = os.path.join(TRAIN_DIR, "reddit_posts_comments_npd.csv")
TRAIT_LEXICON_PATH = os.path.join(DATA_DIR, "npd_trait_lexicons.json")
PARTNER_HINTS_PATH = os.path.join(DATA_DIR, "partner_hints.json")
MANIPULAATION_LEXICON_PATH = os.path.join(DATA_DIR, "npd_manipulation_lexicons.json")

# 建议用环境变量管理密钥
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "Fyrbj-NlMGINKdcBH6Tdww")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "kVAZEWSTdiYEWk0xjY0mUJIgFMW8Sw")
REDDIT_UA = os.getenv("REDDIT_UA", "HamoAI/0.1 by u/chrischengzh")

def load_trait_lexicon(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("npd_trait_lexicons.json 必须是 {trait: [regex,...]} 结构")
    return data

# ---------- 词典：NPD 特征关键词（从外置文件加载并编译） ----------
_raw_lexicon = load_trait_lexicon(TRAIT_LEXICON_PATH)
re_TRAITS = {k: [re.compile(p, re.I) for p in v] for k, v in _raw_lexicon.items()}

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

# 识别“配偶/伴侣”指称的启发式（从 data/partner_hints.json 读取）
PARTNER_HINTS = _load_partner_hints(PARTNER_HINTS_PATH)
re_PARTNER = [re.compile(p, re.I) for p in PARTNER_HINTS]

def has_partner_anchor(text: str) -> bool:
    return any(rp.search(text) for rp in re_PARTNER)

def lexicon_scores(text: str) -> dict:
    scores = {}
    for trait, patterns in re_TRAITS.items():
        hit = sum(1 for rp in patterns if rp.search(text))
        if hit > 0:
            scores[trait] = min(1.0, 0.2 * hit)  # 简单打分：每命中一条+0.2（可调）
    return scores

def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"http\S+|\[.*?\]\(.*?\)", " ", s)  # 链接/markdown
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Zero-shot 多标签 ----------
# 移除 LABEL_TO_DESC 相关定义

def build_zero_shot():
    # 英文社区 → 英文零样本模型
    return pipeline(
        "zero-shot-classification",
        # model="facebook/bart-large-mnli",  # 英文 NLI 强基线
        model="valhalla/distilbart-mnli-12-1",  # 更轻的模型
        device=0,  # 改成GPU 测试下来快一倍多
        tokenizer_kwargs={"truncation": True, "max_length": POST_MAX_LENGTH}
    )

# --------------- 从词典构建 zero-shot 候选与映射 ---------------
def _build_zero_shot_labels_from_lexicon(json_path: str):
    """
    从词典文件（traits/manipulations）提取候选描述标签列表，以及 描述→标准键 的映射。
    统一使用英文标签 label.en（若缺失则回退到键名 k）。
    描述优先级用于作为 zero-shot 的候选文本：desc.en > label.en > desc.zh > label.zh
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return [], {}
    labels = []
    desc2key = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            # 候选描述文本优先用英文
            desc_en = (v.get("desc") or {}).get("en")
            label_en = (v.get("label") or {}).get("en")
            desc_zh = (v.get("desc") or {}).get("zh")
            label_zh = (v.get("label") or {}).get("zh")
            chosen_desc = desc_en or label_en or desc_zh or label_zh
            if not chosen_desc:
                continue
            labels.append(chosen_desc)
            # 标准键统一使用英文显示名；若没有 label.en 则回退到字典键 k
            standard_key = (v.get("label") or {}).get("en") or k
            desc2key[chosen_desc] = standard_key
    return labels, desc2key

# --------------- zero-shot 计算（traits / manipulations） ---------------
def _zero_shot_scores_with_labels(zs, text: str, candidate_labels: list, desc2key: dict) -> dict:
    # 修复语法错误：逗号应为 and
    if (not zs) or (not text) or len(text) < 8 or (not candidate_labels):
        return {}
    try:
        out = zs(text[:POST_MAX_LENGTH], candidate_labels, multi_label=True)
    except Exception:
        return {}
    scores = {}
    try:
        for label, score in zip(out.get("labels", []), out.get("scores", [])):
            k = desc2key.get(label, label)
            scores[k] = float(score)
    except Exception:
        pass
    return scores

def zero_shot_scores(zs, text: str) -> dict:
    """
    traits 的零样本概率，候选从 data/npd_trait_lexicons.json 构建（统一输出 label.en）
    """
    labels, d2k = _build_zero_shot_labels_from_lexicon(TRAIT_LEXICON_PATH)
    return _zero_shot_scores_with_labels(zs, text, labels, d2k)

def zero_shot_scores_manip(zs, text: str) -> dict:
    """
    manipulations 的零样本概率，候选从 data/npd_manipulation_lexicons.json 构建（统一输出 label.en）
    """
    labels, d2k = _build_zero_shot_labels_from_lexicon(MANIPULAATION_LEXICON_PATH)
    return _zero_shot_scores_with_labels(zs, text, labels, d2k)

def fuse_scores(lex_scores: dict, zs_scores: dict, w_lex=0.4, w_zs=0.6) -> dict:
    keys = set(lex_scores) | set(zs_scores)
    fused = {}
    for k in keys:
        fused[k] = w_lex * lex_scores.get(k, 0.0) + w_zs * zs_scores.get(k, 0.0)
    return fused

def partner_filtered(text: str) -> bool:
    # 有“配偶/伴侣”锚点才进行提取（降低误报）
    return has_partner_anchor(text)

# ---------- 抓取 ----------
def fetch_posts_comments():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_UA,
        check_for_updates=False  # 避免额外的 PyPI 请求
    )
    sub = reddit.subreddit(SUBREDDIT)
    rows = []

    # Posts
    for i, post in enumerate(sub.hot(limit=POST_LIMIT), start=1):
        title = clean_text(post.title)
        body = clean_text(post.selftext)
        rows.append({
            "id": post.id,
            "kind": "post",
            "author": str(post.author) if post.author else "[deleted]",
            "created_utc": int(post.created_utc),
            "title": title,
            "body": body,
            "score": post.score,
            "permalink": f"https://www.reddit.com{post.permalink}"
        })
        # 评论（按需控制数量）
        post.comments.replace_more(limit=0)
        taken = 0
        for c in post.comments.list():
            if taken >= MAX_COMMENTS:
                break
            rows.append({
                "id": c.id,
                "kind": "comment",
                "author": str(c.author) if c.author else "[deleted]",
                "created_utc": int(getattr(c, "created_utc", 0)) or 0,
                "title": "",
                "body": clean_text(getattr(c, "body", "")),
                "score": getattr(c, "score", 0),
                "permalink": f"https://www.reddit.com{getattr(c, 'permalink', '')}"
            })
            taken += 1
        time.sleep(0.2)  # 轻微限速，友好一点

    return pd.DataFrame(rows)

def extract_traits(df: pd.DataFrame, zs):
    out_rows = []
    for _, r in df.iterrows():
        text = " ".join([r.get("title", ""), r.get("body", "")]).strip()
        if not text:
            continue
        # 不再基于配偶锚点过滤，全部记录
        # if not partner_filtered(text):
        #     continue

        # 词典（traits）打分（若词典键为英文则可直接融合；否则仅作为辅助可选）
        lex = lexicon_scores(text)

        # traits zero-shot（输出英文键）
        zs_scores_traits = zero_shot_scores(zs, text[:1000])

        # 融合（保持原有两路融合逻辑）
        fused = fuse_scores(lex, zs_scores_traits)

        # manipulations zero-shot（输出英文键）
        manip_scores = zero_shot_scores_manip(zs, text)

        out_rows.append({
            "kind": r["kind"],
            "permalink": r["permalink"],
            "text": text,
            "traits": sorted([(k, round(v, 3)) for k, v in fused.items()], key=lambda x: -x[1]),
            "manipulations": sorted([(k, round(v, 3)) for k, v in manip_scores.items()], key=lambda x: -x[1]),
        })
    return pd.DataFrame(out_rows)

def main():
    print(f"\nFetching from r/{SUBREDDIT} ...")
    t1 = time.time()
    df = fetch_posts_comments()
    t2 = time.time()
    diff_seconds = round(t2 - t1, 2)
    print("总耗时：", diff_seconds, "秒")
    print(f"Fetched rows: {len(df)}")

    print("\nLoading zero-shot NLI model ...")
    t1 = time.time()
    zs = build_zero_shot()
    t2 = time.time()
    diff_seconds = round(t2 - t1, 2)
    print("总耗时：", diff_seconds, "秒")

    print("\nExtracting NPD-like traits ...")
    t1 = time.time()
    result = extract_traits(df, zs)
    t2 = time.time()
    diff_seconds = round(t2 - t1, 2)
    print("总耗时：", diff_seconds, "秒")

    # 合并保存：原始 + 抽取结果
    df.to_csv(OUT_REDDIT_POSTS_NPD_CSV.replace(".csv", "_raw.csv"), index=False)
    result.to_csv(OUT_REDDIT_POSTS_NPD_CSV, index=False)
    print(f"Saved: {OUT_REDDIT_POSTS_NPD_CSV} & {OUT_REDDIT_POSTS_NPD_CSV.replace('.csv', '_raw.csv')}")
    print("Sample:")
    print(result.head(5))

if __name__ == "__main__":
    main()