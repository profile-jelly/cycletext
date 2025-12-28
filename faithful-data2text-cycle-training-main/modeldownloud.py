# import os
# import requests
# from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
#
# def check_huggingface_connectivity():
#     """检测是否能访问 Hugging Face 主站"""
#     try:
#         r = requests.get("https://huggingface.co", timeout=5)
#         return r.status_code == 200
#     except Exception:
#         return False
#
#
# def load_or_download_model(model_name="t5-base", local_dir=r"G:\abnormal\cycletext\model"):
#     """
#     自动检测模型是否存在，如果不存在则下载或提示。
#     model_name: 模型名或Hugging Face路径，例如 "t5-base"
#     local_dir: 指定本地目录，如 r"G:\\models\\t5-base"
#     """
#     print(f"🔍 Checking model: {model_name}")
#
#     # 若指定本地路径
#     if local_dir and os.path.exists(local_dir):
#         print(f"✅ Found local model at: {local_dir}")
#         tokenizer = AutoTokenizer.from_pretrained(local_dir)
#         model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
#         return tokenizer, model
#
#     # 自动检测缓存
#     from transformers.utils import cached_file
#     try:
#         cached_path = cached_file(model_name, "config.json")
#         if cached_path:
#             print(f"✅ Model already cached at: {os.path.dirname(cached_path)}")
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#             return tokenizer, model
#     except Exception:
#         pass
#
#     # 若缓存不存在，检测网络
#     print("⚠️ Model not found locally, checking internet access...")
#     if check_huggingface_connectivity():
#         print("🌐 Hugging Face reachable, downloading model...")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         print("✅ Download complete, model cached for future use.")
#         return tokenizer, model
#     else:
#         print("❌ Cannot reach Hugging Face. Please manually download model from:")
#         print(f"   👉 https://huggingface.co/{model_name}")
#         print("   Then place it under a local directory and re-run with:")
#         print('   load_or_download_model(local_dir=r"G:\\models\\t5-base")')
#         return None, None
#
#
# if __name__ == "__main__":
#     # 修改为你的目标模型
#     model_name = "t5-base"
#     local_path = r"G:\models\t5-base"
#
#     tokenizer, model = load_or_download_model(model_name, local_dir=local_path)
#
#     if model is not None:
#         print("\n✅ Model loaded successfully!")
#         text = "Studies have shown that owning a dog is good for you"
#         inputs = tokenizer("summarize: " + text, return_tensors="pt")
#         outputs = model.generate(**inputs, max_new_tokens=30)
#         print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretradine(r"G:\models\t5-base")
model = T5ForConditionalGeneration.from_pretrained(r"G:\models\t5-base")

inputs = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)

print("✅ Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
import os
# import shutil
# from pathlib import Path
#
# # 源路径（你本机缓存的模型）
# src = Path(r"C:\Users\gh\.cache\huggingface\hub\models--t5-base\snapshots\a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1")
#
# # 目标路径
# dst = Path(r"G:\models\t5-base")
#
# print(f"🔍 Source: {src}")
# print(f"📁 Target: {dst}")
#
# if not src.exists():
#     raise FileNotFoundError(f"❌ 源模型文件夹不存在: {src}")
#
# dst.mkdir(parents=True, exist_ok=True)
#
# # 拷贝文件
# for file in src.iterdir():
#     if file.is_file():
#         shutil.copy(file, dst / file.name)
#         print(f"✅ Copied: {file.name}")
#
# print("\n🎯 模型已成功复制到本地目录。")
# print(f"→ 本地路径: {dst}")
# print("\n你现在可以在代码中使用以下路径加载模型：\n")
# print(f'  from transformers import T5Tokenizer, T5ForConditionalGeneration')
# print(f'  tokenizer = T5Tokenizer.from_pretrained(r"{dst}")')
# print(f'  model = T5ForConditionalGeneration.from_pretrained(r"{dst}")')
