import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # 或 "Agg"

import matplotlib.pyplot as plt

# 读取 json
# with open("data2text_results.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
with open("test2data_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# 评价指标
# metrics = ["BLEU", "METEOR", "ROUGE_L", "BERTScore_F1"]
metrics = ["Triple_P", "Triple_R", "Triple_F1", "Exact_Match"]
# 选取模型
# models = [
#     "data2text_100",
#     "data2text_full",
#     "data2text_cycle_9"
# ]
# models = [
#     "text2data_100",
#     "text2data_full",
#     "text2data_cycle_9"
# ]
# models = [
#      "data2text_cycle_9",
#      "lora_data2text_full",
#      "lora_data2text_decoder",
#      "lora_aware_data2text_decoder"
# ]
# models = [
#      "text2data_cycle_9",
#      "lora_text2data_full",
#      "lora_text2data_decoder",
#      "lora_aware_text2data_decoder"
# ]
# models = [
#     "data2text_cycle_9",
#     "lora_aware_data2text_decoder",
#     "data2text_full"
# ]
models = [
    "text2data_cycle_9",
    "lora_text2data_decoder",
    "text2data_full"
]
# 按指标组织数据
values = {
    metric: [data[m][metric] for m in models]
    for metric in metrics
}

x = np.arange(len(metrics))
width = 0.22

plt.figure(figsize=(10, 5))

for i, model in enumerate(models):
    plt.bar(
        x + i * width,
        [values[metric][i] for metric in metrics],
        width,
        label=model
    )

plt.xticks(x + width * (len(models) - 1) / 2, metrics)
plt.ylabel("Score")
# plt.title("Data2Text Performance Comparison by Metric")
plt.title("Text2Data Performance Comparison by Metric")
plt.legend()
plt.tight_layout()
plt.show()
