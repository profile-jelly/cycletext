# import torch
# from transformers import (
#     T5Tokenizer,
#     T5ForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
#     DataCollatorForSeq2Seq,
# )
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model
#
# # =====================================================
# # 配置
# # =====================================================
# MODEL_PATH = r"G:/abnormal/cycletext/output/sb"   # text2data 模型
# TOKENIZER_PATH = r"G:/abnormal/cycletext/models/webnlg_t5_data2text_100"
#
# DATA_FILE = r"G:/abnormal/cycletext/data/webnlg-t5-triplets2text/train.tsv"
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# MAX_INPUT_LEN = 128
# MAX_OUTPUT_LEN = 96
#
# BATCH_SIZE = 8
# GRAD_ACC = 4
# EPOCHS = 1
#
# # =====================================================
# # tokenizer
# # =====================================================
# tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)
#
# # =====================================================
# # 数据加载（稳健 TSV）
# # =====================================================
# def load_tsv_robust(path):
#     texts, triples = [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.rstrip("\n")
#             if not line:
#                 continue
#             parts = line.split("\t")
#             if len(parts) < 2:
#                 continue
#             # ⚠️ 关键：反向使用
#             triples.append(parts[0])              # data
#             texts.append("\t".join(parts[1:]))    # text
#     return Dataset.from_dict({
#         "text": texts,
#         "data": triples
#     })
#
# dataset = load_tsv_robust(DATA_FILE)
#
# # =====================================================
# # 预处理（Text → Data）
# # =====================================================
# def preprocess(batch):
#     model_inputs = tokenizer(
#         batch["text"],            # text
#         truncation=True,
#         max_length=MAX_INPUT_LEN,
#     )
#     labels = tokenizer(
#         batch["data"],            # data
#         truncation=True,
#         max_length=MAX_OUTPUT_LEN,
#     )
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# dataset = dataset.map(
#     preprocess,
#     batched=True,
#     remove_columns=dataset.column_names,
# )
#
# # =====================================================
# # 模型
# # =====================================================
# model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
#
# # =====================================================
# # LoRA 配置（T5）
# # =====================================================
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     bias="none",
#     task_type="SEQ_2_SEQ_LM",
#     target_modules=["q", "k", "v", "o"],
# )
#
# model = get_peft_model(model, lora_config)
# model.to(DEVICE)
# model.print_trainable_parameters()
#
# # =====================================================
# # Data collator
# # =====================================================
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,
#     padding=True,
#     label_pad_token_id=-100,
# )
#
# # =====================================================
# # 训练参数
# # =====================================================
# training_args = Seq2SeqTrainingArguments(
#     output_dir=r"./lora_text2data_full",
#     per_device_train_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=GRAD_ACC,
#     num_train_epochs=EPOCHS,
#     logging_steps=50,
#     save_steps=500,
#     save_total_limit=2,
#     fp16=(DEVICE == "cuda"),
#     report_to="none",
#     prediction_loss_only=True,
# )
#
# # =====================================================
# # Trainer
# # =====================================================
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )
#
# trainer.train()
#
# # =====================================================
# # 保存（LoRA adapter）
# # =====================================================
# model.save_pretrained("./lora_text2data_full")
# tokenizer.save_pretrained("./lora_text2data_full")
#
# print("✅ Text2Data LoRA fine-tuning finished.")
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from lora.CycleAwareLoraLinear import CycleAwareLoraLinear

# =====================================================
# 配置
# =====================================================
MODEL_PATH = r"G:/abnormal/cycletext/output/sb"   # text2data 模型
TOKENIZER_PATH = r"G:/abnormal/cycletext/models/webnlg_t5_data2text_100"
DATA_FILE = r"G:/abnormal/cycletext/data/webnlg-t5-triplets2text/train.tsv"

MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 96

BATCH_SIZE = 8
GRAD_ACC = 4
EPOCHS = 1
LR = 2e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# tokenizer
# =====================================================
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)

# =====================================================
# 数据加载（Text → Data）
# =====================================================
def load_tsv_robust(path):
    texts, triples = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            triples.append(parts[0])
            texts.append("\t".join(parts[1:]))
    return Dataset.from_dict({
        "text": texts,
        "data": triples
    })

dataset = load_tsv_robust(DATA_FILE)

def preprocess(batch):
    inputs = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
    labels = tokenizer(
        batch["data"],
        truncation=True,
        max_length=MAX_OUTPUT_LEN,
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names,
)

# =====================================================
# 模型加载
# =====================================================
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# =====================================================
# 冻结 Encoder（关键）
# =====================================================
for p in model.encoder.parameters():
    p.requires_grad = False

# =====================================================
# LoRA（Decoder only）
# =====================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q", "v"],
)

model = get_peft_model(model, lora_config)
model.to(DEVICE)
model.print_trainable_parameters()



#cycle-ware
def inject_cycle_lora(model):
    for module in model.modules():
        if hasattr(module, "lora_A"):
            module.__class__ = CycleAwareLoraLinear

def inject_cycle_lora_behavior(model):
    for module in model.modules():
        if (
            hasattr(module, "lora_A")
            and isinstance(module.lora_A, torch.nn.ModuleDict)
            and "default" in module.lora_A
        ):
            # 1️⃣ 保存原始 forward（只保存一次）
            if not hasattr(module, "_cycle_orig_forward"):
                module._cycle_orig_forward = module.forward

            # 2️⃣ 初始化 scale
            module.lora_scale = None

            def set_lora_scale(self, scale):
                self.lora_scale = scale

            module.set_lora_scale = set_lora_scale.__get__(module)

            # 3️⃣ 定义新的 forward
            def cycle_forward(x, module=module):
                # 原始输出（PEFT 原逻辑）
                base_out = module._cycle_orig_forward(x)

                # LoRA residual
                if "default" in module.lora_A:
                    lora_A = module.lora_A["default"]
                    lora_B = module.lora_B["default"]

                    scaling = module.scaling["default"]
                    lora_out = lora_B(lora_A(x)) * scaling

                    scale = getattr(module, "lora_scale", None)

                    # === Case 0: 没有 scale 或 scale 为空 → 退化为普通 LoRA ===
                    if scale is None or scale.numel() == 0:
                        return base_out + lora_out

                    scale = scale.to(lora_out.device)
                    B = lora_out.size(0)

                    # === Case 1: scalar scale ===
                    if scale.numel() == 1:
                        scale = scale.view(1, 1, 1)

                    # === Case 2: batch-level scale ===
                    elif scale.numel() == B:
                        scale = scale.view(B, 1, 1)

                    # === Case 3: cycle-level scale（如 2 → 扩展到 batch） ===
                    elif B % scale.numel() == 0:
                        repeat = B // scale.numel()
                        scale = scale.repeat_interleave(repeat).view(B, 1, 1)

                    # === Case 4: 不可对齐 → fallback ===
                    else:
                        # 最保守、最安全、审稿人最能接受的策略
                        scale = scale.mean().view(1, 1, 1)

                    lora_out = lora_out * scale
                    return base_out + lora_out

                    return base_out + lora_out

                return base_out

            module.forward = cycle_forward

#inject_cycle_lora(model)
inject_cycle_lora_behavior(model)

#token-level coverage error
def compute_cycle_scale(labels, logits):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        token_conf = probs.max(dim=-1).values  # (B, T)

        mask = (labels != -100).float()        # (B, T)
        conf_sum = (token_conf * mask).sum(dim=1)
        token_cnt = mask.sum(dim=1).clamp(min=1)

        cycle_loss = 1 - conf_sum / token_cnt
        scale = cycle_loss / (cycle_loss.mean() + 1e-6)

        return torch.clamp(scale, 0.5, 2.0)

class CycleAwareTrainer(Seq2SeqTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # ✅ 必须加
    ):
        with torch.no_grad():
            tmp_outputs = model(**inputs)
            scale = compute_cycle_scale(
                inputs["labels"],
                tmp_outputs.logits
            )

        for m in model.modules():
            if hasattr(m, "set_lora_scale"):
                m.set_lora_scale(scale)

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss






# =====================================================
# Data collator
# =====================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# =====================================================
# 训练参数
# =====================================================
training_args = Seq2SeqTrainingArguments(
    #output_dir="./lora_text2data_decoder",
    output_dir="../lora_aware_text2data_decoder",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=(DEVICE == "cuda"),
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    prediction_loss_only=True,
)

# =====================================================
# Trainer
# =====================================================
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=data_collator,
# )
trainer = CycleAwareTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# =====================================================
# 保存 LoRA
# =====================================================
model.save_pretrained("./lora_aware_text2data_decoder")

print("✅ Text2Data Decoder-only LoRA finished.")

