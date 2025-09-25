import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM

# -----------------------
# Step 1. 判别式模型 (Erlangshen)
# -----------------------
sentiment_model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"  # 轻量版
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

# -----------------------
# Step 2. 生成式大模型 (Qwen)
# -----------------------
gen_model_name = "Qwen/Qwen1.5-1.8B-Chat"  # 建议小模型跑 Demo，7B 会爆内存
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, device_map="auto")

def explain_trigger(sentence, candidate_labels):
    """用生成模型解释触发原因"""
    prompt = f"""
请分析下面句子可能引发对话冲突或情绪升级的原因。候选标签如下：
{candidate_labels}

句子：{sentence}

请按以下格式输出：
可能的触发原因：
- 标签 (置信度或合理性说明)
"""
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    outputs = gen_model.generate(**inputs, max_new_tokens=256)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------
# Step 3. Demo 流程
# -----------------------
conversation = [
    "我觉得自己好可怜，天天一个人加班。",
    "你他妈的闭嘴！"
]

# 1) 判别式模型逐句情绪分类
results = sentiment_analyzer(conversation)

print("🟡 情绪检测结果：")
for i, (sent, res) in enumerate(zip(conversation, results)):
    print(f"- 第{i+1}句: {sent} → {res['label']} ({res['score']:.2f})")

# 2) 触发点检测（简单版：看最后一句 vs 前面句子的差异）
last_res = results[-1]
trigger_idx, max_shift = -1, 0
for i in range(len(results)-1):
    shift = last_res['score'] - results[i]['score'] if results[i]['label'] != "NEGATIVE" else 0
    if shift > max_shift:
        max_shift, trigger_idx = shift, i

# 3) 如果找到触发点 → 调用生成式大模型解释
if trigger_idx >= 0:
    trigger_sentence = conversation[trigger_idx]
    print(f"\n⚠️ 触发点可能是第 {trigger_idx+1} 句: \"{trigger_sentence}\"")

    candidate_labels = [
    "责备或指责",
    "语气不耐烦",
    "缺乏关心或支持",
    "表达无奈和抱怨",
    "表达模糊，容易被误解",
    "带有批评意味"
  ]
    explanation = explain_trigger(trigger_sentence, candidate_labels)
    print("\n🔎 可能的触发原因：")
    print(explanation)
else:
    print("\n⚠️ 未能找到明显触发点。")