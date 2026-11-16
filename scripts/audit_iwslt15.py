
from pathlib import Path
from itertools import zip_longest
from collections import Counter
import json
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "local_artifacts/data/iwslt15/nmt-en-vi-master/data"
)

OUTPUT_DIR = ROOT / "results/data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": ("train.en", "train.vi"),
    "validation": ("tst2012.en", "tst2012.vi"),
    "test": ("tst2013.en", "tst2013.vi"),
}


def normalize(text):
    """Chuan hoa Unicode NFC, khong doi chu hoa."""
    return unicodedata.normalize("NFC", text.strip())


def audit_split(name, en_file, vi_file):
    en_path = DATA_DIR / en_file
    vi_path = DATA_DIR / vi_file

    if not en_path.exists() or not vi_path.exists():
        raise FileNotFoundError(
            f"Missing dataset files: {en_path}, {vi_path}"
        )

    stats = Counter()
    seen_pairs = set()
    valid_pairs = set()

    with en_path.open("r", encoding="utf-8") as f_en, \
         vi_path.open("r", encoding="utf-8") as f_vi:

        for en, vi in zip_longest(f_en, f_vi):
            stats["total"] += 1

            if en is None or vi is None:
                stats["unmatched"] += 1
                continue

            en = normalize(en)
            vi = normalize(vi)

            if not en:
                stats["empty_en"] += 1

            if not vi:
                stats["empty_vi"] += 1

            if not en or not vi:
                continue

            # Kiem tra theo so token tach bang khoang trang.
            # Chua phai SentencePiece token.
            en_len = len(en.split())
            vi_len = len(vi.split())

            if en_len > 100 or vi_len > 100:
                stats["too_long"] += 1

            ratio = max(en_len, vi_len) / max(
                min(en_len, vi_len), 1
            )

            if ratio > 2.5:
                stats["bad_length_ratio"] += 1

            pair = (en, vi)

            if pair in seen_pairs:
                stats["duplicate_pairs"] += 1

            seen_pairs.add(pair)
            valid_pairs.add(pair)

    return dict(stats), valid_pairs


def main():
    report = {}
    split_pairs = {}

    for name, files in SPLITS.items():
        stats, pairs = audit_split(name, *files)
        report[name] = stats
        split_pairs[name] = pairs

        print(f"\n=== {name.upper()} ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

    print("\n=== CROSS-SPLIT OVERLAP ===")

    checks = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for a, b in checks:
        overlap = split_pairs[a] & split_pairs[b]
        key = f"{a}_vs_{b}"
        report[key] = len(overlap)
        print(f"{key}: {len(overlap)}")

    output_path = OUTPUT_DIR / "iwslt15_audit.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            report, f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
