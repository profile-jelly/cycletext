import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

def load_model(model_cfg, tokenizer_path, device):
    tokenizer = T5Tokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True
    )

    if model_cfg["type"] == "full":
        model = T5ForConditionalGeneration.from_pretrained(
            model_cfg["path"],
            local_files_only=True
        )

    elif model_cfg["type"] == "lora":
        base = T5ForConditionalGeneration.from_pretrained(
            model_cfg["base"],
            local_files_only=True
        )
        model = PeftModel.from_pretrained(
            base,
            model_cfg["path"],
            local_files_only=True
        )
    else:
        raise ValueError("Unknown model type")

    model.eval().to(device)
    return tokenizer, model
