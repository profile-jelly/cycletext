# import argparse
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
# #from datasets import load_dataset
# from evaluate import load
# import os
# def load_t5_dataset(source_path, target_path, tokenizer, max_input_length=256, max_output_length=256):
#     """加载并编码 source/target 文本"""
#     data = {"source": open(source_path, encoding="utf-8").read().splitlines(),
#             "target": open(target_path, encoding="utf-8").read().splitlines()}
#     dataset = {"train": [{"input_text": s, "target_text": t} for s, t in zip(data["source"], data["target"])]}
#
#     def preprocess_function(batch):
#         model_inputs = tokenizer(batch["input_text"], max_length=max_input_length, truncation=True)
#         labels = tokenizer(text_target=batch["target_text"], max_length=max_output_length, truncation=True)
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
#
#     from datasets import Dataset
#     dataset = Dataset.from_list(dataset["train"])
#     tokenized_dataset = dataset.map(preprocess_function, batched=True)
#     return tokenized_dataset
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--source_file", type=str, required=True)
#     parser.add_argument("--target_file", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--t5_model", type=str, default="t5-base")
#     args = parser.parse_args()
#
#     print("🚀 加载模型与分词器...")
#     tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
#     model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
#
#     print("📘 加载训练数据...")
#     dataset = load_t5_dataset(args.source_file, args.target_file, tokenizer)
#
#     print("🎯 开始训练...")
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         learning_rate=3e-4,
#         per_device_train_batch_size=8,
#         num_train_epochs=3,
#         logging_dir=os.path.join(args.output_dir, "logs"),
#         save_total_limit=1,
#         logging_steps=100
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         tokenizer=tokenizer
#     )
#
#     trainer.train()
#     trainer.save_model(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     print(f"✅ 模型已保存到: {args.output_dir}")
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
import os

def load_t5_dataset(source_path, target_path, tokenizer, max_input_length=256, max_output_length=256):
    """加载并编码 source/target 文本"""
    data = {
        "source": open(source_path, encoding="utf-8").read().splitlines(),
        "target": open(target_path, encoding="utf-8").read().splitlines()
    }
    dataset = [{"input_text": s, "target_text": t} for s, t in zip(data["source"], data["target"])]

    def preprocess_function(batch):
        model_inputs = tokenizer(batch["input_text"], max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=max_output_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_list(dataset)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True)
    parser.add_argument("--target_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--t5_model", type=str, default="t5-base")
    args = parser.parse_args()

    print("加载模型与分词器...")
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)

    print(" 加载训练数据...")
    dataset = load_t5_dataset(args.source_file, args.target_file, tokenizer)

    #  使用 DataCollatorForSeq2Seq 自动处理 padding 和 labels
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        #truncation=True
    )

    print("开始训练...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=1,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator  #  注意这里
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f" 模型已保存到: {args.output_dir}")
