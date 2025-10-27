# file: reddit_npds_extract.py
import os
import re
import time
import pandas as pd
import praw
from transformers import pipeline

# ---------- 配置 ----------
SUBREDDIT = "NarcissisticSpouses"
POST_LIMIT = 30       # 抓取帖子数
MAX_COMMENTS = 300    # 评论上限（总量控制）
OUT_CSV = "ns_posts_comments_npds.csv"

# 建议用环境变量管理密钥
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "Fyrbj-NlMGINKdcBH6Tdww")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "kVAZEWSTdiYEWk0xjY0mUJIgFMW8Sw")
REDDIT_UA = os.getenv("REDDIT_UA", "HamoAI/0.1 by u/chrischengzh")

# ---------- 词典：NPD 特征关键词（可扩充） ----------
TRAIT_LEXICON = {
    "grandiosity": [
        r"\b(grandiose|superior|perfect|flawless|never wrong)\b",
        r"\b(think(s)? (he|she|they) (is|are) (above|better) (everyone|others))\b"
    ],
    "fantasy_of_success_power_beauty": [
        r"\b(fantasi(e|es)|delusion(s)? of (grandeur|success|power|beauty))\b",
        r"\b(ideal love|soulmate fantasy)\b"
    ],
    "special_unique": [
        r"\b(unique|special|exceptional|too (smart|important))\b"
    ],
    "need_for_admiration": [
        r"\b(admir(e|ation)|praise|constant validation|applause)\b"
    ],
    "entitlement": [
        r"\b(entitled|deserve(s|d)? everything|rules (don'?t|do not) apply)\b"
    ],
    "exploitative": [
        r"\b(exploit(s|ed|ing)?|use(s|d)? me|manipulat(e|ive|ion))\b"
    ],
    "lack_of_empathy": [
        r"\b(lack(s)? of empathy|no empathy|doesn'?t care|no remorse)\b"
    ],
    "envious_arrogant": [
        r"\b(arrogant|smug|condescending|envious|jealous)\b"
    ],
    # 操控行为
    "love_bombing": [
        r"\b(love ?bomb(ing)?|shower(ed)? me with (gifts|attention|affection))\b"
    ],
    "gaslighting": [
        r"\b(gaslight(ing|ed)?|you('?| )re crazy|made me doubt my reality)\b"
    ],
    "triangulation": [
        r"\b(triangulat(e|ion)|compare(d)? me to (ex|others)|brought a third person)\b"
    ],
    "devaluation_discard": [
        r"\b(devalu(e|ation)|discard(ed)?|sudden devalue|put me down)\b"
    ],
    "silent_treatment": [
        r"\b(silent treatment|stonewall(ing)?|ignore(d|s|ing) me)\b"
    ],
    "blame_shifting": [
        r"\b(blame shifting|it'?s all my fault|never their fault|always blame me)\b"
    ],
    "future_faking": [
        r"\b(future faking|made promises (he|she|they) never kept)\b"
    ],
    "hoovering": [
        r"\b(hoover(ing)?|pull(ed)? me back|suck(ed)? me back)\b"
    ]
}

# 识别“配偶/伴侣”指称的启发式
PARTNER_HINTS = [
    r"\b(my|our)\s+(husband|wife|spouse|partner|bf|boyfriend|gf|girlfriend)\b",
    r"\b(ex[- ]?(husband|wife|partner|bf|boyfriend|gf|girlfriend))\b",
    r"\b(he|she|his|her)\b"  # 简化：后续与上下文合并
]

re_PARTNER = [re.compile(p, re.I) for p in PARTNER_HINTS]
re_TRAITS = {k: [re.compile(p, re.I) for p in v] for k, v in TRAIT_LEXICON.items()}

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
# 把标签写成自然语言描述，便于 NLI 模型判别
LABEL_TO_DESC = {
    "grandiosity": "shows grandiosity or superiority",
    "fantasy_of_success_power_beauty": "has fantasies of unlimited success, power, beauty or ideal love",
    "special_unique": "believes they are special and unique",
    "need_for_admiration": "needs excessive admiration",
    "entitlement": "has a sense of entitlement",
    "exploitative": "is interpersonally exploitative",
    "lack_of_empathy": "lacks empathy",
    "envious_arrogant": "is envious or arrogant",
    "love_bombing": "uses love bombing",
    "gaslighting": "gaslights others",
    "triangulation": "uses triangulation",
    "devaluation_discard": "devalues or discards partners",
    "silent_treatment": "uses silent treatment",
    "blame_shifting": "engages in blame shifting",
    "future_faking": "does future faking",
    "hoovering": "does hoovering"
}
ZERO_SHOT_LABELS = list(LABEL_TO_DESC.values())
DESC2KEY = {v: k for k, v in LABEL_TO_DESC.items()}

def build_zero_shot():
    # 英文社区 → 英文零样本模型
    return pipeline(
        "zero-shot-classification",
        # model="facebook/bart-large-mnli",  # 英文 NLI 强基线
        model="valhalla/distilbart-mnli-12-1",  # 更轻的模型
        device=-1,  # 用 CPU（更稳；MPS 对 NLI 不一定更快）
        tokenizer_kwargs={"truncation": True, "max_length": 384}  # 控制长度
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
        # time.sleep(0.2)  # 轻微限速，友好一点

    return pd.DataFrame(rows)

def extract_traits(df: pd.DataFrame, zs):
    out_rows = []
    for _, r in df.iterrows():
        text = " ".join([r.get("title", ""), r.get("body", "")]).strip()
        if not text:
            continue
        if not partner_filtered(text):
            # 没有配偶锚点时，可选择跳过或仍做检测，这里保守跳过
            continue
        lex = lexicon_scores(text)
        zs_scores = zero_shot_scores(zs, text[:1000])  # 控制长度以提速
        fused = fuse_scores(lex, zs_scores)
        # 设定阈值（可按验证集调优）
        picks = {k: v for k, v in fused.items() if v >= 0.45}
        out_rows.append({
            "kind": r["kind"],
            "permalink": r["permalink"],
            "text": text[:2000],
            "lexicon_hits": sorted([(k, round(v, 3)) for k, v in lex.items()], key=lambda x: -x[1]),
            "zero_shot_top": sorted([(k, round(v, 3)) for k, v in zs_scores.items()], key=lambda x: -x[1])[:5],
            "traits": sorted([(k, round(v, 3)) for k, v in picks.items()], key=lambda x: -x[1])
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
    df.to_csv(OUT_CSV.replace(".csv", "_raw.csv"), index=False)
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} & {OUT_CSV.replace('.csv', '_raw.csv')}")
    print("Sample:")
    print(result.head(5))

if __name__ == "__main__":
    main()