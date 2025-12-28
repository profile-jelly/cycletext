# metrics_text.py
import evaluate

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def eval_text_metrics(preds, refs):
    assert len(preds) == len(refs), \
        f"Preds ({len(preds)}) != Refs ({len(refs)})"

    # BLEU expects List[List[str]]
    refs_bleu = [[r] for r in refs]

    results = {}

    results["BLEU"] = bleu.compute(
        predictions=preds,
        references=refs_bleu
    )["bleu"]

    results["METEOR"] = meteor.compute(
        predictions=preds,
        references=refs
    )["meteor"]

    results["ROUGE_L"] = rouge.compute(
        predictions=preds,
        references=refs,
        use_aggregator=True,
    )["rougeL"]

    bert = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="en",
        rescale_with_baseline=True,
    )

    results["BERTScore_F1"] = sum(bert["f1"]) / len(bert["f1"])

    return results

