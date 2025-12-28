import re

def normalize_token(x):
    return x.strip().lower()

def parse_triples(triple_str):
    triples = set()

    if not triple_str or not triple_str.strip():
        return triples

    # 统一分隔符：| ; || 换成 |
    triple_str = re.sub(r"\s*\|\s*|\s*;\s*", "|", triple_str)

    for t in triple_str.split("|"):
        t = t.strip()

        # 去掉括号
        t = t.strip("()")

        # 兼容中英文逗号
        parts = re.split(r"\s*,\s*", t)

        if len(parts) != 3:
            continue

        s, p, o = map(normalize_token, parts)
        triples.add((s, p, o))

    return triples

def eval_triple_metrics(preds, refs):
    tp = fp = fn = em = 0

    for p, r in zip(preds, refs):
        p_set = parse_triples(p)
        r_set = parse_triples(r)

        tp += len(p_set & r_set)
        fp += len(p_set - r_set)
        fn += len(r_set - p_set)

        if p_set == r_set:
            em += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "Triple_P": precision,
        "Triple_R": recall,
        "Triple_F1": f1,
        "Exact_Match": em / len(preds),
    }
