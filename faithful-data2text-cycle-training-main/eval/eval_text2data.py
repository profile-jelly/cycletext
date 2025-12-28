from eval_config import *
from load_model import load_model
from webnlg_dataset import load_webnlg_tsv
from generate import generate_text
from metrics_triple import eval_triple_metrics
from utils import save_results
def run():
    inputs, refs = load_webnlg_tsv(
        TEST_FILE,
        direction="text2data"
    )
    all_results = {}
    for name, cfg in TEXT2DATA_MODELS.items():
        print(f"\n▶ Evaluating {name}")
        tokenizer, model = load_model(cfg, TOKENIZER_PATH, DEVICE)

        preds = generate_text(
            model,
            tokenizer,
            inputs,
            MAX_OUTPUT_LEN,
            DEVICE,
            batch_size=8,
            num_beams=1,  # ❗
        )

        scores = eval_triple_metrics(preds, refs)
        all_results[name] = scores
        print(scores)
    save_results(all_results, "test2data_aware_results.json")


if __name__ == "__main__":
    run()
