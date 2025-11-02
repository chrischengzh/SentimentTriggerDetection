import praw
import csv
import json
import time
from datetime import datetime, timezone   # ✅ 改成 timezone.utc

# 配置 Reddit API
reddit = praw.Reddit(
    client_id="Fyrbj-NlMGINKdcBH6Tdww",
    client_secret="kVAZEWSTdiYEWk0xjY0mUJIgFMW8Sw",
    user_agent="HamoAI/0.1 by u/chrischengzh"
)

subreddit = reddit.subreddit("NarcissisticSpouses")

# 存储结果
posts_data = []

t1 = time.time()
for post in subreddit.hot(limit=50):  # 先取 50 条
    post.comments.replace_more(limit=0)  # 展开评论
    comments_data = []
    for comment in post.comments.list()[:10]:  # 每帖取前 10 条评论
        comments_data.append({
            "id": comment.id,
            "author": str(comment.author),
            "score": comment.score,
            "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
            "body": comment.body
        })

    post_info = {
        "id": post.id,
        "title": post.title,
        "author": str(post.author),
        "score": post.score,
        "upvote_ratio": post.upvote_ratio,
        "num_comments": post.num_comments,
        "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
        "link": f"https://reddit.com{post.permalink}",
        "selftext": post.selftext,
        "comments": comments_data
    }

    posts_data.append(post_info)
t2 = time.time()
diff_seconds = round(t2 - t1, 2)
print("\n抓取Reddit总耗时：", diff_seconds, "秒")

# ===== 导出 CSV（简化版） =====
csv_file = "training/reddit_posts.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id","title","author","score","num_comments","created_utc","link","selftext"])
    writer.writeheader()
    for post in posts_data:
        writer.writerow({
            "id": post["id"],
            "title": post["title"],
            "author": post["author"],
            "score": post["score"],
            "num_comments": post["num_comments"],
            "created_utc": post["created_utc"],
            "link": post["link"],
            "selftext": post["selftext"][:500]  # 截取正文前 500 字，避免过长
        })

print(f"CSV 导出完成: {csv_file}")

# ===== 导出 JSON（完整数据） =====
json_file = "training/reddit_posts.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(posts_data, f, ensure_ascii=False, indent=2)

print(f"JSON 导出完成: {json_file}")