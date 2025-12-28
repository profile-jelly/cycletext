# dataset.py
import csv
from typing import Optional
def load_webnlg_tsv(
    path,
    direction: str = "data2text",
    max_samples: Optional[int] = None,
):
    """
    Load WebNLG TSV file safely for both directions.

    Args:
        path (str): path to .tsv file
        direction (str): "data2text" or "text2data"
        max_samples (int, optional): limit number of samples

    Returns:
        inputs (List[str])
        targets (List[str])
    """

    if direction not in {"data2text", "text2data"}:
        raise ValueError(
            f"direction must be 'data2text' or 'text2data', got: {direction}"
        )

    inputs = []
    targets = []

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for i, row in enumerate(reader):
            # 跳过非法行
            if len(row) < 2:
                continue

            data = row[0].strip()
            text = row[1].strip()

            if not data or not text:
                continue

            if direction == "data2text":
                inp, tgt = data, text
            else:  # text2data
                inp, tgt = text, data

            inputs.append(inp)
            targets.append(tgt)

            if max_samples is not None and len(inputs) >= max_samples:
                break

    assert len(inputs) == len(targets), "Inputs and targets length mismatch"

    return inputs, targets
