import praw
from datetime import datetime

reddit = praw.Reddit(
    client_id="Fyrbj-NlMGINKdcBH6Tdww",          # personal use script 字段
    client_secret="kVAZEWSTdiYEWk0xjY0mUJIgFMW8Sw",  # secret 字段
    user_agent="HamoAI/0.1 by u/chrischengzh"
)

# 测试能否访问
subreddit = reddit.subreddit("NarcissisticSpouses")
for post in subreddit.hot(limit=5):
    print("="*50)
    print(f"Title: {post.title}")
    print(f"Author: {post.author}")
    print(f"Score (Upvotes): {post.score}")
    print(f"Upvote Ratio: {post.upvote_ratio}")
    print(f"Comments Count: {post.num_comments}")
    print(f"Created: {datetime.utcfromtimestamp(post.created_utc)}")
    print(f"Link: https://reddit.com{post.permalink}")
    print(f"Content: {post.selftext[:200]}...")  # 截取前200字

    # 获取评论（按热度排序）
    post.comments.replace_more(limit=0)  # 展开 "MoreComments"
    top_comments = post.comments.list()[:5]  # 取前5条评论
    print("\nTop Comments:")
    for comment in top_comments:
        print(f"- {comment.author}: ({comment.score} upvotes) {comment.body[:100]}...")
