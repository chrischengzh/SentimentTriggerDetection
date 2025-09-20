# V1.0.0
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import time
from datetime import datetime

# =========================
# 设备选择
# =========================
# 自动检测是否支持 Apple Silicon GPU (MPS)
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
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
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
explain_model_name = "facebook/bart-large-mnli"
explain_tokenizer = AutoTokenizer.from_pretrained(explain_model_name)
explain_model = AutoModelForSequenceClassification.from_pretrained(explain_model_name)
explain_model.to(device)

# zero-shot classification pipeline
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
    results = [sentiment_analyzer(utt)[0] for utt in conversation]

    print("\n情绪分析结果：")
    for i, (utt, res) in enumerate(zip(conversation, results)):
        print(f"{i+1}. {utt} --> {res}")

    last_res = results[-1]
    if last_res['label'] != "NEGATIVE":
        print("\n❌ 最后一句不是负面情绪，不需要触发检测。")
        return None, None

    trigger_idx = -1
    max_shift = 0
    for i in range(len(results)-1):
        shift = last_res['score'] - results[i]['score'] if results[i]['label'] != "NEGATIVE" else 0
        if shift > max_shift:
            max_shift = shift
            trigger_idx = i

    if trigger_idx >= 0:
        trigger_sentence = conversation[trigger_idx]
        print(f"\n⚠️ 触发点可能是第 {trigger_idx+1} 句: \"{trigger_sentence}\"")

        # 候选解释标签
        candidate_labels = [
            "责备或指责",
            "语气不耐烦",
            "缺乏关心或支持",
            "表达模糊，容易被误解",
            "带有批评意味"
        ]
        explanation = explain_analyzer(trigger_sentence, candidate_labels)
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
t1 = time.time()
if __name__ == "__main__":
    convo = [
        "Hey Cathy, could you help me with my tax refund?",
        "I already told you, that's not my problem.",
        "Why are you being so rude?"
    ]
    detect_trigger(convo)
t2 = time.time()
diff_seconds = round(t2 - t1, 2)
print("\n总耗时：", diff_seconds, "秒")
