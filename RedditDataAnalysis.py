import praw

reddit = praw.Reddit(
    client_id="Fyrbj-NlMGINKdcBH6Tdww",          # personal use script 字段
    client_secret="kVAZEWSTdiYEWk0xjY0mUJIgFMW8Sw",  # secret 字段
    user_agent="HamoAI/0.1 by u/chrischengzh"
)

# 测试能否访问
subreddit = reddit.subreddit("NarcissisticSpouses")
for post in subreddit.new(limit=5):
    print(post.title)