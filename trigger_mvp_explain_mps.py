# V1.0.2
import time
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# =========================
# 场景触发词典:加载family, work, study...
# =========================
def load_labels(config_path, domain="family"):
    with open(config_path, "r", encoding="utf-8") as f:
        all_labels = json.load(f)
    return all_labels.get(domain, [])

# =========================
# 设备选择:自动检测是否支持 Apple Silicon GPU (MPS)
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_id = 0  # pipeline 里用 0 表示启用 GPU
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_id = 0
    print("Using NVIDIA CUDA GPU")
else:
    device = torch.device("cpu")
    device_id = -1
    print("Using CPU")

# =========================
# Sentiment Analysis (情绪分析)
# =========================
sentiment_model_name = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model.to(device)

# sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    device=device_id
)

# =========================
# Zero-shot Classification (解释层)
# =========================
explain_model_name = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI"
explain_tokenizer = AutoTokenizer.from_pretrained(explain_model_name)
explain_model = AutoModelForSequenceClassification.from_pretrained(explain_model_name)
explain_model.to(device)

# zero-shot classification pipeline 用零样本分类模型分析，这句话可能“触发”的原因
explain_analyzer = pipeline(
    "zero-shot-classification",
    model=explain_model,
    tokenizer=explain_tokenizer,
    device=device_id
)

# =========================
# Trigger Detection 函数
# =========================
def detect_trigger(conversation):
    t1 = time.time()
    results = [sentiment_analyzer(utt)[0] for utt in conversation]
    t2 = time.time()
    diff_seconds = round(t2 - t1, 2)
    print("\nSentiment Analysis耗时：", diff_seconds, "秒")
    print("\n情绪分析结果：")
    for i, (utt, res) in enumerate(zip(conversation, results)):
        print(f"{i+1}. {utt} --> {res}")

    last_res = results[-1]
    if last_res['label'] != "Negative":
        print("\n❌ 最后一句不是负面情绪，不需要触发检测。")
        return None, None

    trigger_idx = -1    # 初始化触发点的索引，-1 表示还没找到
    max_shift = 0   # 保存目前发现的最大“情绪分数变化”
    for i in range(len(results)-1): # 遍历 results 列表（对话每一句的情绪分析结果），到倒数第二句
        # 计算“情绪分数变化”
        shift = last_res['score'] - results[i]['score'] if results[i]['label'] != "NEGATIVE" else 0
        if shift > max_shift:   # 如果这次的变化更大
            max_shift = shift   # 更新最大变化
            trigger_idx = i # 记录触发点的位置

    if trigger_idx >= 0:
        trigger_sentence = conversation[trigger_idx]    # 对话里触发最后情绪爆发的关键句
        print(f"\n⚠️ 触发点可能是第 {trigger_idx+1} 句: \"{trigger_sentence}\"")

        candidate_labels = load_labels("data/trigger_labels.json", domain="family")
        t1 = time.time()
        explanation = explain_analyzer(trigger_sentence, candidate_labels)
        t2 = time.time()
        diff_seconds = round(t2 - t1, 2)
        print("\nTrigger Detection耗时：", diff_seconds, "秒")
        print("\n🔎 可能的触发原因：")
        for lbl, score in zip(explanation["labels"], explanation["scores"]):
            print(f"- {lbl} ({score:.2f})")

        return trigger_sentence, explanation
    else:
        print("\n⚠️ 未能找到明显触发点。")
        return None, None


# =========================
# 测试运行
# =========================
if __name__ == "__main__":
    convo_english = [
        "Hey Cathy, could you help me with my tax refund?",
        "I already told you, that's not my problem.",
        "Why are you being so rude?"
    ]
    convo_chinese = [
        "Cathy：其实好多都说不通的，这里也有人suv工签延期被拒的，不知道这个怎么弄。你觉得呢？",
        "我：有不合理，可以申诉或重新提交申请，但还是要看审核官和运气。没有一条100%的路径。",
        "Cathy：我也可以不申请工签对我而言没有意义。",
        '我：这个你可以再考虑一下，有决定了我们开个股东会议。',
        "Cathy：我工签被拒和股东大会有啥关联，要不找人把我替了，省得那么多麻烦，旅游签简简单单，小孩读大学我也可以回去了。",
        "Cathy：都来看我笑话，我运气差。"
    ]
    detect_trigger(convo_chinese)