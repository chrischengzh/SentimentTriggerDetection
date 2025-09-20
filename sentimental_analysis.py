# V 1.0.0
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

# 加载模型
model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

# 构建 pipeline
sentiment = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device = device_id
)

# 测试的多段对话（你可以随便改成自己的句子）
dialogues = [
	"Cathy：其实好多都说不通的，这里也有人suv工签延期被拒的，不知道这个怎么弄。你觉得呢？",
	"我：有不合理，可以申诉或重新提交申请，但还是要看审核官和运气。没有一条100%的路径。",
	"Cathy：我也可以不申请工签对我而言没有意义。",
	"我：这个你可以再考虑一下，有决定了我们开个股东会议。",
	"Cathy：我工签被拒和股东大会有啥关联，要不找人把我替了，省得那么多麻烦，旅游签简简单单，小孩读大学我也可以回去了。",
	"Cathy：都来看我笑话，我运气差。"
]

t1 = time.time()

# 对每句话做情绪识别
print("情绪识别结果：")
for i, sentence in enumerate(dialogues, 1):
    result = sentiment(sentence)[0]  # [{'label': 'positive', 'score': 0.95}]
    label = result['label']
    score = round(result['score'], 4)
    print(f"{i}. {sentence} -> {label} (score={score})")

t2 = time.time()
diff_seconds = round(t2 - t1, 2)
print("\n总耗时：", diff_seconds, "秒")
