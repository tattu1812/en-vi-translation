
from pathlib import Path
from itertools import zip_longest

DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data/iwslt15/nmt-en-vi-master/data"
)

SPLITS = {
    "train": ("train.en", "train.vi"),
    "validation": ("tst2012.en", "tst2012.vi"),
    "test": ("tst2013.en", "tst2013.vi"),
}


def check_split(name, en_file, vi_file):
    en_path = DATA_DIR / en_file
    vi_path = DATA_DIR / vi_file

    total = 0
    empty_en = 0
    empty_vi = 0
    unmatched = 0

    with en_path.open(encoding="utf-8") as f_en, \
         vi_path.open(encoding="utf-8") as f_vi:

        for en, vi in zip_longest(f_en, f_vi):
            total += 1

            if en is None or vi is None:
                unmatched += 1
                continue

            if not en.strip():
                empty_en += 1

            if not vi.strip():
                empty_vi += 1

    print(f"\n=== {name.upper()} ===")
    print(f"Total lines: {total}")
    print(f"Empty English: {empty_en}")
    print(f"Empty Vietnamese: {empty_vi}")
    print(f"Unmatched lines: {unmatched}")


if __name__ == "__main__":
    print("DATA DIR:", DATA_DIR)

    for name, (en_file, vi_file) in SPLITS.items():
        check_split(name, en_file, vi_file)
