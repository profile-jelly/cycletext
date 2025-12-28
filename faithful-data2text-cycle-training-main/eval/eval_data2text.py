from eval_config import *
from load_model import load_model
from webnlg_dataset import load_webnlg_tsv
from generate import generate_text
from metrics_text import eval_text_metrics
from utils import save_results
def run():
    inputs, refs = load_webnlg_tsv(
        TEST_FILE,
        direction="data2text"
    )

    all_results = {}

    for name, cfg in DATA2TEXT_MODELS.items():
        print(f"\n▶ Evaluating {name}")

        tokenizer, model = load_model(cfg, TOKENIZER_PATH, DEVICE)
        model.eval()

        preds = generate_text(
            model,
            tokenizer,
            inputs,
            MAX_OUTPUT_LEN,
            DEVICE,
            batch_size=16,
            num_beams=1,
        )

        scores = eval_text_metrics(preds, refs)
        all_results[name] = scores

        print(scores)

    save_results(all_results, "data2text_aware_results.json")


if __name__ == "__main__":
    run()
