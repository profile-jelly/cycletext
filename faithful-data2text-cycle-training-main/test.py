# import torch
# from tqdm import tqdm
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from evaluate import load
# from nltk.tokenize import word_tokenize
#
# # =========================================================
# # 🔧 直接写死的配置（已按你给的路径填写）
# # =========================================================
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TOKENIZER_PATH = r"G:\abnormal\cycletext\models\webnlg_t5_data2text_100"
#
# # ===== 选择要测试的模型 =====
# # 改这里即可切换模型
# MODEL_PATH = r"G:\abnormal\cycletext\output\text2data-10"
# # MODEL_PATH = r"G:\abnormal\cycletext\output\data2text-10"
#
# # ===== 对应测试集 =====
# TEST_FILE = r"G:\abnormal\cycletext\data\processed\webnlg-t5-unpaired\texts_unpaired.txt"
# # TEST_FILE = r"G:\abnormal\cycletext\data\processed\webnlg-t5-triplets2text\test.tsv"
#
# BATCH_SIZE = 8
# MAX_INPUT_LENGTH = 128
# MAX_OUTPUT_LENGTH = 128
# NUM_BEAMS = 1
#
# # =========================================================
# # 🚀 加载模型
# # =========================================================
#
# print("Loading model...")
# tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
# model.to(DEVICE)
# model.eval()
#
# # =========================================================
# # 📄 加载测试数据
# # =========================================================
#
# print("Loading test dataset...")
# dataset = load_dataset(
#     "csv",
#     data_files={"test": TEST_FILE},
#     delimiter="\t",
#     column_names=["source", "target"]
# )
#
# def tokenize(batch):
#     return tokenizer(
#         batch["source"],
#         padding="max_length",
#         truncation=True,
#         max_length=MAX_INPUT_LENGTH
#     )
#
# dataset = dataset.map(tokenize, batched=True)
# dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "source", "target"]
# )
#
# loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)
#
# # =========================================================
# # 📊 指标
# # =========================================================
#
# metric_bleu = load("bleu")
# metric_meteor = load("meteor")
# metric_rouge = load("rouge")
#
# predictions = []
# references = []
#
# # =========================================================
# # 🔍 推理
# # =========================================================
#
# print("Running inference...")
# with torch.no_grad():
#     for batch in tqdm(loader):
#         outputs = model.generate(
#             input_ids=batch["input_ids"].to(DEVICE),
#             attention_mask=batch["attention_mask"].to(DEVICE),
#             max_length=MAX_OUTPUT_LENGTH,
#             num_beams=NUM_BEAMS,
#             early_stopping=True
#         )
#
#         preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         predictions.extend(preds)
#         references.extend(batch["target"])
#
# # =========================================================
# # 🧮 计算指标
# # =========================================================
#
# bleu = metric_bleu.compute(
#     predictions=[word_tokenize(p) for p in predictions],
#     references=[[word_tokenize(r)] for r in references]
# )["bleu"]
#
# meteor = metric_meteor.compute(
#     predictions=[" ".join(word_tokenize(p)) for p in predictions],
#     references=[[" ".join(word_tokenize(r))] for r in references]
# )["meteor"]
#
# rouge = metric_rouge.compute(predictions=predictions, references=references)
#
# # =========================================================
# # ✅ 输出结果
# # =========================================================
#
# print("\n================ Test Result ================")
# print(f"Model Path : {MODEL_PATH}")
# print(f"Test File  : {TEST_FILE}")
# print(f"BLEU       : {bleu:.4f}")
# print(f"METEOR     : {meteor:.4f}")
# print(f"ROUGE-L    : {rouge['rougeL'].mid.fmeasure:.4f}")
# print("============================================")
#
# # =========================================================
# # 💾 保存生成结果
# # =========================================================
#
# out_path = MODEL_PATH + ".test.generations.txt"
# with open(out_path, "w", encoding="utf-8") as f:
#     for p in predictions:
#         f.write(p + "\n")
#
# print(f"\nGenerated outputs saved to:\n{out_path}")
import random
import random
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import word_tokenize
from evaluate import load
from pathlib import Path

# ======================================================
# 配置区
# ======================================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# tokenizer 永远用最初的 t5-base
TOKENIZER_PATH = Path(r"G:/abnormal/cycletext/models/webnlg_t5_data2text_100")

# ===== 选择模型（2 选 1）=====
MODEL_PATH = Path(r"G:/abnormal/cycletext/output/sb")
TASK_TYPE = "text2data"   # 或 "data2text"

# ===== 数据文件 =====
# paired（可算 BLEU）
# TEST_FILE = Path(r"G:/abnormal/cycletext/data/processed/webnlg-t5-triplets2text/test.tsv")

# unpaired（只能 sanity check）
TEST_FILE = Path(r"G:/abnormal/cycletext/data/processed/webnlg-t5-unpaired/texts_unpaired.txt")

MAX_SAMPLES = 50
MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 64
# ======================================================

print("Loading tokenizer & model...")
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
model.eval()

print("Loading data...")

# ---------- 根据文件类型加载 ----------
if TEST_FILE.suffix == ".tsv":
    dataset = load_dataset(
        "csv",
        data_files={"test": str(TEST_FILE)},
        delimiter="\t",
        column_names=["source", "target"]
    )["test"]
    HAS_TARGET = True
else:
    dataset = load_dataset(
        "text",
        data_files={"test": str(TEST_FILE)}
    )["test"]
    HAS_TARGET = False

# 随机抽样
dataset = dataset.shuffle(seed=42).select(range(min(MAX_SAMPLES, len(dataset))))

if HAS_TARGET:
    bleu = load("bleu")
    predictions, references = [], []

print("\nRunning quick evaluation...\n")

used = 0
for i, sample in enumerate(dataset):

    if HAS_TARGET:
        source = sample["source"]
        target = sample["target"]

        if target is None or not isinstance(target, str) or target.strip() == "":
            continue
    else:
        source = sample["text"]
        target = None

    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_LEN
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_OUTPUT_LEN,
            num_beams=1
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    used += 1

    # BLEU（只有 paired 数据才算）
    if HAS_TARGET:
        predictions.append(word_tokenize(pred))
        references.append([word_tokenize(target)])

    # 打印前 5 条
    if used <= 5:
        print("=" * 80)
        print(f"[{used}] INPUT:")
        print(source)
        print("\nPRED:")
        print(pred)
        if HAS_TARGET:
            print("\nGOLD:")
            print(target)

# ---------- 结果 ----------
print("\n" + "=" * 80)
print(f"Effective samples used: {used}")

if HAS_TARGET:
    bleu_score = bleu.compute(
        predictions=predictions,
        references=references
    )["bleu"]
    print(f"Quick BLEU: {bleu_score:.4f}")
else:
    print("Unpaired data → qualitative sanity check only (no BLEU).")

print("=" * 80)

