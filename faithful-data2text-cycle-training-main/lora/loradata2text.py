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
# MODEL_PATH = r"G:/abnormal/cycletext/output/data2text-9"
# # ↑ 一定是 data2text 基座模型
#
# TOKENIZER_PATH = r"G:/abnormal/cycletext/models/webnlg_t5_data2text_100"
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
# # 数据加载（Data → Text，正向）
# # =====================================================
# def load_tsv_robust(path):
#     triples, texts = [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.rstrip("\n")
#             if not line:
#                 continue
#             parts = line.split("\t")
#             if len(parts) < 2:
#                 continue
#
#             triples.append(parts[0])              # data
#             texts.append("\t".join(parts[1:]))    # text
#
#     return Dataset.from_dict({
#         "data": triples,
#         "text": texts
#     })
#
# dataset = load_tsv_robust(DATA_FILE)
#
# # =====================================================
# # 预处理（Data → Text）
# # =====================================================
# def preprocess(batch):
#     model_inputs = tokenizer(
#         batch["data"],          # ✅ data 作为输入
#         truncation=True,
#         max_length=MAX_INPUT_LEN,
#     )
#     labels = tokenizer(
#         batch["text"],          # ✅ text 作为输出
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
# model.to(DEVICE)
#
# # =====================================================
# # LoRA 配置（Full LoRA，与你原代码一致）
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
#     output_dir=r"./lora_data2text_full",
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
# model.save_pretrained("./lora_data2text_full")
# tokenizer.save_pretrained("./lora_data2text_full")
#
# print("✅ Data2Text LoRA fine-tuning finished.")
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

# =====================================================
# 配置
# =====================================================
MODEL_PATH = r"G:/abnormal/cycletext/output/data2text-9"
# ↑ data2text 方向的基础模型（一定是 data2text）

TOKENIZER_PATH = r"G:/abnormal/cycletext/models/webnlg_t5_data2text_100"
DATA_FILE = r"G:/abnormal/cycletext/data/webnlg-t5-triplets2text/train.tsv"

MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 128

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
# 数据加载（Data → Text，正向）
# =====================================================
def load_tsv_robust(path):
    triples, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            triples.append(parts[0])              # data
            texts.append("\t".join(parts[1:]))    # text
    return Dataset.from_dict({
        "data": triples,
        "text": texts
    })

dataset = load_tsv_robust(DATA_FILE)

# =====================================================
# 预处理（Data → Text）
# =====================================================
def preprocess(batch):
    inputs = tokenizer(
        batch["data"],        # ✅ data 作为输入
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
    labels = tokenizer(
        batch["text"],        # ✅ text 作为输出
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
# 模型加载（Data2Text）
# =====================================================
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# =====================================================
# 冻结 Encoder（CycleText 推荐）
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
    target_modules=["q", "v"],   # T5 decoder attention
)


def inject_cycle_lora_behavior(model):
    for module in model.modules():
        if (
            hasattr(module, "lora_A")
            and isinstance(module.lora_A, torch.nn.ModuleDict)
            and "default" in module.lora_A
        ):
            if not hasattr(module, "_cycle_orig_forward"):
                module._cycle_orig_forward = module.forward

            module.lora_scale = None

            def set_lora_scale(self, scale):
                self.lora_scale = scale

            module.set_lora_scale = set_lora_scale.__get__(module)

            def cycle_forward(x, module=module):
                base_out = module._cycle_orig_forward(x)

                if "default" in module.lora_A:
                    lora_A = module.lora_A["default"]
                    lora_B = module.lora_B["default"]
                    scaling = module.scaling["default"]

                    lora_out = lora_B(lora_A(x)) * scaling
                    scale = getattr(module, "lora_scale", None)

                    if scale is None or scale.numel() == 0:
                        return base_out + lora_out

                    scale = scale.to(lora_out.device)
                    B = lora_out.size(0)

                    if scale.numel() == 1:
                        scale = scale.view(1, 1, 1)
                    elif scale.numel() == B:
                        scale = scale.view(B, 1, 1)
                    elif B % scale.numel() == 0:
                        scale = scale.repeat_interleave(B // scale.numel()).view(B, 1, 1)
                    else:
                        scale = scale.mean().view(1, 1, 1)

                    return base_out + lora_out * scale

                return base_out

            module.forward = cycle_forward

model = get_peft_model(model, lora_config)
inject_cycle_lora_behavior(model)
model.to(DEVICE)
model.print_trainable_parameters()

# =====================================================
# Data collator
# =====================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)
def compute_cycle_scale(labels, logits):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        token_conf = probs.max(dim=-1).values

        mask = (labels != -100).float()
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
        num_items_in_batch=None,
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
# 训练参数
# =====================================================
training_args = Seq2SeqTrainingArguments(
    output_dir="../lora_aware_data2text_decoder",
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
# 保存 LoRA adapter
# =====================================================
model.save_pretrained("./lora_aware_data2text_decoder")

print("✅ Data2Text Decoder-only LoRA finished.")
