import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 测试集
# ===============================
TEST_FILE = r"G:/abnormal/cycletext/data/webnlg-t5-triplets2text/test.tsv"

# ===============================
# tokenizer（统一）
# ===============================
TOKENIZER_PATH = r"G:/abnormal/cycletext/models/webnlg_t5_data2text_100"

# ===============================
# Data → Text 模型
# ===============================
DATA2TEXT_MODELS = {
    # "data2text_100": {
    #     "type": "full",
    #     "path": r"G:\abnormal\cycletext\models\webnlg_t5_data2text_100",
    # },
    # "data2text_full": {
    #     "type": "full",
    #     "path": r"G:\abnormal\cycletext\models\webnlg_t5_data2text_full",
    # },
    # "data2text_9": {
    #     "type": "full",
    #     "path": r"G:\abnormal\cycletext\output\data2text-9",
    # },
    # "lora_data2text_full": {
    #     "type": "lora",
    #     "base": r"G:\abnormal\cycletext\output\data2text-9",
    #     "path": r"G:\abnormal\cycletext\faithful-data2text-cycle-training-main\faithful-data2text-cycle-training-main\lora_data2text_full",
    # },
    # "lora_data2text_decoder": {
    #     "type": "lora",
    #     "base": r"G:\abnormal\cycletext\output\data2text-9",
    #     "path": r"G:\abnormal\cycletext\faithful-data2text-cycle-training-main\faithful-data2text-cycle-training-main\lora_data2text_decoder",
    # },
    "lora_aware_data2text_decoder": {
        "type": "lora",
        "base": r"G:\abnormal\cycletext\output\data2text-9",
        "path": r"G:\abnormal\cycletext\faithful-data2text-cycle-training-main\faithful-data2text-cycle-training-main\lora_aware_data2text_decoder",
    },
}

# ===============================
# Text → Data 模型
# ===============================
TEXT2DATA_MODELS = {
    # "text2data_100": {
    #     "type": "full",
    #     "path": r"G:\abnormal\cycletext\models\webnlg_t5_text2data_100",
    # },
    # "text2data_full": {
    #     "type": "full",
    #     "path":r"G:\abnormal\cycletext\models\webnlg_t5_text2data_full",
    # },
    # "text2data_9": {
    #     "type": "full",
    #     "path": r"G:\abnormal\cycletext\output\sb",
    # },
    # "lora_text2data_full": {
    #     "type": "lora",
    #     "base": r"G:\abnormal\cycletext\output\sb",
    #     "path": r"G:\abnormal\cycletext\faithful-data2text-cycle-training-main\faithful-data2text-cycle-training-main\lora_text2data_full",
    # },
    # "lora_text2data_decoder": {
    #     "type": "lora",
    #     "base": r"G:\abnormal\cycletext\output\sb",
    #     "path": r"G:\abnormal\cycletext\faithful-data2text-cycle-training-main\faithful-data2text-cycle-training-main\lora_text2data_decoder",
    # },
    "lora_aware_text2data_decoder": {
        "type": "lora",
        "base": r"G:\abnormal\cycletext\output\sb",
        "path": r"G:\abnormal\cycletext\faithful-data2text-cycle-training-main\faithful-data2text-cycle-training-main\lora_aware_text2data_decoder",
    },
}

MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 128
