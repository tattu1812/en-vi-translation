
from pathlib import Path
import argparse
import json

from sacrebleu.metrics import BLEU


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "shared_assets/data/iwslt15"

LENGTH_GROUPS = {
    "le_10": (1, 10),
    "11_20": (11, 20),
    "21_30": (21, 30),
    "gt_30": (31, float("inf")),
}


def load_lines(path):
    """Doc file, moi dong ung voi mot cau."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    return path.read_text(
        encoding="utf-8"
    ).splitlines()


def calculate_bleu(predictions, references):
    """
    Tinh corpus BLEU bang SacreBLEU.

    predictions: list[str] - ban dich cua model
    references: list[str] - ban dich tham chieu
    """
    if len(predictions) != len(references):
        raise ValueError(
            "So cau du doan khac so cau tham chieu"
        )

    if len(predictions) == 0:
        raise ValueError("Danh sach cau rong")

    bleu = BLEU(
        tokenize="13a",
        lowercase=False,
        smooth_method="exp",
        effective_order=False,
    )

    result = bleu.corpus_score(
        predictions,
        [references]
    )

    return {
        "bleu": round(result.score, 4),
        "num_sentences": len(predictions),
        "signature": str(bleu.get_signature()),
    }


def calculate_bleu_by_length(
    sources,
    predictions,
    references
):
    """
    Chia theo so don vi tach bang khoang trang
    cua cau nguon tieng Anh.
    """
    if not (
        len(sources)
        == len(predictions)
        == len(references)
    ):
        raise ValueError(
            "Source, prediction, reference khac so dong"
        )

    results = {}

    for group_name, (min_len, max_len) in (
        LENGTH_GROUPS.items()
    ):
        indices = [
            i
            for i, src in enumerate(sources)
            if min_len <= len(src.split()) <= max_len
        ]

        if not indices:
            results[group_name] = {
                "bleu": None,
                "num_sentences": 0,
            }
            continue

        group_predictions = [
            predictions[i] for i in indices
        ]

        group_references = [
            references[i] for i in indices
        ]

        results[group_name] = calculate_bleu(
            group_predictions,
            group_references
        )

    return results


def evaluate(sources, predictions, references):
    """Danh gia tong the va theo do dai."""
    if not (
        len(sources)
        == len(predictions)
        == len(references)
    ):
        raise ValueError("Khong khop so luong cau")

    if not sources:
        raise ValueError("Du lieu danh gia rong")

    if any(not x.strip() for x in sources):
        raise ValueError("Co cau nguon rong")

    if any(not x.strip() for x in references):
        raise ValueError("Co cau tham chieu rong")

    return {
        "overall": calculate_bleu(
            predictions, references
        ),
        "by_length": calculate_bleu_by_length(
            sources, predictions, references
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Danh gia dich may bang SacreBLEU"
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="File ban dich model, moi dong mot cau"
    )

    parser.add_argument(
        "--sources",
        default=str(DATA_DIR / "test.en")
    )

    parser.add_argument(
        "--references",
        default=str(DATA_DIR / "test.vi")
    )

    parser.add_argument(
        "--output",
        default="results/bleu/evaluation.json"
    )

    args = parser.parse_args()

    sources = load_lines(args.sources)
    predictions = load_lines(args.predictions)
    references = load_lines(args.references)

    result = evaluate(
        sources, predictions, references
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== SACREBLEU EVALUATION ===")
    print("Corpus BLEU:", result["overall"]["bleu"])
    print("Signature:", result["overall"]["signature"])

    print("\n=== BLEU BY SOURCE LENGTH ===")
    for group, stats in result["by_length"].items():
        print(
            f"{group}: BLEU={stats['bleu']}, "
            f"n={stats['num_sentences']}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
