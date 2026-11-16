
from pathlib import Path
from itertools import zip_longest
from collections import Counter
import json
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT / "local_artifacts/data/iwslt15"
    / "nmt-en-vi-master/data"
)

CLEAN_DIR = (
    ROOT / "local_artifacts/data/iwslt15/cleaned"
)

REPORT_DIR = ROOT / "results/data"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": ("train.en", "train.vi"),
    "validation": ("tst2012.en", "tst2012.vi"),
    "test": ("tst2013.en", "tst2013.vi"),
}

MAX_LENGTH = 100
MAX_RATIO = 2.5


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def read_pairs(en_name, vi_name):
    en_path = RAW_DIR / en_name
    vi_path = RAW_DIR / vi_name

    pairs = []

    with en_path.open(encoding="utf-8") as f_en, \
         vi_path.open(encoding="utf-8") as f_vi:

        for line_no, (en, vi) in enumerate(
            zip_longest(f_en, f_vi), start=1
        ):
            if en is None or vi is None:
                raise ValueError(
                    f"Unmatched lines at {line_no}"
                )

            pairs.append((
                normalize(en),
                normalize(vi)
            ))

    return pairs


def write_pairs(name, pairs):
    en_path = CLEAN_DIR / f"{name}.en"
    vi_path = CLEAN_DIR / f"{name}.vi"

    with en_path.open("w", encoding="utf-8") as f_en, \
         vi_path.open("w", encoding="utf-8") as f_vi:

        for en, vi in pairs:
            f_en.write(en + "\n")
            f_vi.write(vi + "\n")


def clean_train(train_pairs, protected_pairs):
    stats = Counter()
    cleaned = []
    seen = set()

    for en, vi in train_pairs:
        stats["original"] += 1

        # Moi cap chi duoc gan mot ly do loai chinh.
        if not en or not vi:
            stats["removed_empty"] += 1
            continue

        en_len = len(en.split())
        vi_len = len(vi.split())

        if en_len > MAX_LENGTH or vi_len > MAX_LENGTH:
            stats["removed_too_long"] += 1
            continue

        ratio = max(en_len, vi_len) / min(en_len, vi_len)

        if ratio > MAX_RATIO:
            stats["removed_bad_ratio"] += 1
            continue

        pair = (en, vi)

        if pair in protected_pairs:
            stats["removed_leakage"] += 1
            continue

        if pair in seen:
            stats["removed_duplicate"] += 1
            continue

        seen.add(pair)
        cleaned.append(pair)

    stats["cleaned"] = len(cleaned)
    stats["removed_total"] = (
        stats["original"] - stats["cleaned"]
    )

    return cleaned, dict(stats)


def main():
    datasets = {
        name: read_pairs(*files)
        for name, files in SPLITS.items()
    }

    protected_pairs = (
        set(datasets["validation"])
        | set(datasets["test"])
    )

    clean_train_pairs, stats = clean_train(
        datasets["train"],
        protected_pairs
    )

    # Train duoc loc.
    write_pairs("train", clean_train_pairs)

    # Validation va test giu nguyen so luong.
    for name in ("validation", "test"):
        write_pairs(name, datasets[name])

    # Kiem tra sau khi lam sach.
    assert len(clean_train_pairs) == len(set(clean_train_pairs))
    assert not (set(clean_train_pairs) & protected_pairs)

    report = {
        "settings": {
            "unicode": "NFC",
            "max_whitespace_tokens": MAX_LENGTH,
            "max_length_ratio": MAX_RATIO,
            "train_only_filtering": True,
        },
        "train": stats,
        "validation": len(datasets["validation"]),
        "test": len(datasets["test"]),
        "validation_test_overlap": len(
            set(datasets["validation"])
            & set(datasets["test"])
        ),
    }

    report_path = REPORT_DIR / "iwslt15_cleaning.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== CLEANING REPORT ===")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nValidation: {len(datasets['validation'])}")
    print(f"Test: {len(datasets['test'])}")

    print(f"\nCleaned data: {CLEAN_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
