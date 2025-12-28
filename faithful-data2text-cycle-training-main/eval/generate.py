import torch
from tqdm import tqdm

def generate_text(
    model,
    tokenizer,
    inputs,
    max_len,
    device,
    batch_size=4,      # text2data 建议 1~4
    num_beams=1,       # text2data 不要 beam search
):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch_inputs = inputs[i : i + batch_size]

            enc = tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=384,        # 🔥【关键】硬限制 encoder 输入
                return_tensors="pt",
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_len,  # decoder 输出长度
                num_beams=num_beams,
                do_sample=False,
            )

            batch_preds = tokenizer.batch_decode(
                out,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            preds.extend(batch_preds)

            # ✅ 防止评测跑到一半显存碎片化
            torch.cuda.empty_cache()

    return preds
