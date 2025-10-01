import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM

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

# -----------------------
# Step 1. 判别式模型 (Erlangshen)
# -----------------------
sentiment_model_name = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"  # 轻量版
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
model.to(device)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device_id)

# -----------------------
# Step 2. 生成式大模型 (Qwen)
# -----------------------
gen_model_name = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI"
# gen_model_name = "Qwen/Qwen1.5-1.8B-Chat"  # 建议小模型跑 Demo，7B 会爆内存
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
gen_model.to(device)

import json
def explain_trigger(sentence, candidate_labels):
    """用生成模型解释触发原因，返回 JSON 格式"""
    prompt = f"""
请分析下面句子可能引发对话冲突或情绪升级的原因。候选标签如下：
{candidate_labels}

句子：{sentence}

请严格按以下 JSON 格式输出：
{{
  "triggers": [
    {{
      "label": "标签名",
      "reason": "触发原因的解释",
      "confidence": "高/中/低 或 数值"
    }}
  ]
}}
"""

    # 构建输入
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)

    # 调用大模型生成
    outputs = gen_model.generate(**inputs, max_new_tokens=256)

    # 解码文本
    raw_output = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 尝试解析成 JSON
    try:
        parsed_output = json.loads(raw_output)
        return parsed_output
    except json.JSONDecodeError:
        # 如果解析失败，就返回原始文本，避免报错
        return {"raw_output": raw_output}

# def explain_trigger(sentence, candidate_labels):
#     """用生成模型解释触发原因"""
#     prompt = f"""
# 请分析下面句子可能引发对话冲突或情绪升级的原因。候选标签如下：
# {candidate_labels}
#
# 句子：{sentence}
#
# 请按以下格式输出：
# 可能的触发原因：
# - 标签 (置信度或合理性说明)
# """
#     inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
#     outputs = gen_model.generate(**inputs, max_new_tokens=256)
#     return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------
# Step 3. Demo 流程
# -----------------------
# conversation = [
#     "我觉得自己好可怜，天天一个人加班。",
#     "你他妈的闭嘴！"
# ]
conversation = [
    "Cathy：其实好多都说不通的，这里也有人suv工签延期被拒的，不知道这个怎么弄。你觉得呢？",
    "我：有不合理，可以申诉或重新提交申请，但还是要看审核官和运气。没有一条100%的路径。",
    "Cathy：我也可以不申请工签对我而言没有意义。",
    '我：这个你可以再考虑一下，有决定了我们开个股东会议。',
    "Cathy：我工签被拒和股东大会有啥关联，要不找人把我替了，省得那么多麻烦，旅游签简简单单，小孩读大学我也可以回去了。",
    "Cathy：都来看我笑话，我运气差。"
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
    # print(json.dumps(explanation, indent=2, ensure_ascii=False))
    # 如果模型返回了结构化 triggers
    if "triggers" in explanation:
        print("可能的触发原因：")
        for trig in explanation["triggers"]:
            label = trig.get("label", "未知标签")
            reason = trig.get("reason", "")
            confidence = trig.get("confidence", "")
            print(f"- {label} ({confidence})：{reason}")
    else:
        # 如果解析失败，返回原始输出
        print("模型原始输出：")
        print(explanation.get("raw_output", explanation))
else:
    print("\n⚠️ 未能找到明显触发点。")