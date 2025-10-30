import os
import re
import time
import pandas as pd
import praw
from transformers import pipeline
import json

# ---------- 配置 ----------
SUBREDDIT = "NarcissisticSpouses"
POST_LIMIT = 10       # 抓取帖子数
MAX_COMMENTS = 30    # 评论上限（总量控制）
OUT_CSV = "ns_posts_comments_npds.csv"

# 建议用环境变量管理密钥
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "Fyrbj-NlMGINKdcBH6Tdww")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "kVAZEWSTdiYEWk0xjY0mUJIgFMW8Sw")
REDDIT_UA = os.getenv("REDDIT_UA", "HamoAI/0.1 by u/chrischengzh")

# ---------- 词典：NPD 特征关键词（可扩充） ----------
# 每个键代表一种典型的自恋特征（NPD Trait），
# 每个值是对应的英文关键词 / 表达的正则匹配模式。
DATA_DIR = "data"
TRAIT_LEXICON_PATH = os.path.join(DATA_DIR, "trait_lexicon.json")

def load_trait_lexicon(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("trait_lexicon.json 必须是 {trait: [regex,...]} 结构")
    return data

# ---------- 词典：NPD 特征关键词（从外置文件加载并编译） ----------
_raw_lexicon = load_trait_lexicon(TRAIT_LEXICON_PATH)
re_TRAITS = {k: [re.compile(p, re.I) for p in v] for k, v in _raw_lexicon.items()}

# 识别“配偶/伴侣”指称的启发式
PARTNER_HINTS = [
    r"\b(my|our)\s+(husband|wife|spouse|partner|bf|boyfriend|gf|girlfriend)\b",
    r"\b(ex[- ]?(husband|wife|partner|bf|boyfriend|gf|girlfriend))\b",
    r"\b(he|she|his|her)\b"  # 简化：后续与上下文合并
]
PARTNER_HINTS += [  #增加中文支持
    r"(我(的)?|我们(的)?|咱们(的)?|我家|我们家|咱家)\s*(丈夫|老公|先生|老伴|妻子|老婆|太太|贤内助|男女朋友|男友|女友|男朋友|女朋友|对象|伴侣|配偶|爱人|另一半)",
    r"(前任|前夫|前妻|前男友|前女友|前男朋友|前女朋友|前对象|前伴侣|前配偶)",
    r"(他|她|他的|她的|ta|TA)"
]

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
# 把标签写成自然语言描述，便于 NLI 模型判别
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

def build_zero_shot():
    # 英文社区 → 英文零样本模型
    return pipeline(
        "zero-shot-classification",
        # model="facebook/bart-large-mnli",  # 英文 NLI 强基线
        model="valhalla/distilbart-mnli-12-1",  # 更轻的模型
        device=0,  # 改成GPU 测试下来快一倍多
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