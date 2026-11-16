
from pathlib import Path
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "local_artifacts/data/iwslt15/cleaned"
TOKENIZER_DIR = ROOT / "local_artifacts/tokenizer"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_EN = DATA_DIR / "train.en"
TRAIN_VI = DATA_DIR / "train.vi"

CORPUS_FILE = TOKENIZER_DIR / "train_corpus.txt"
MODEL_PREFIX = TOKENIZER_DIR / "spm_envi_bpe_8000"


def prepare_corpus():
    """Gop hai ngon ngu tu train vao mot corpus."""

    with CORPUS_FILE.open("w", encoding="utf-8") as out:
        for path in (TRAIN_EN, TRAIN_VI):
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    text = line.strip()
                    if text:
                        out.write(text + "\n")

    print(f"Corpus saved: {CORPUS_FILE}")


def train():
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_FILE),
        model_prefix=str(MODEL_PREFIX),
        model_type="bpe",
        vocab_size=8000,

        # Chot ID token dac biet
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        character_coverage=1.0,
        normalization_rule_name="identity",
        hard_vocab_limit=True
    )

    print("Tokenizer training completed!")


def verify():
    model_path = str(MODEL_PREFIX) + ".model"
    sp = spm.SentencePieceProcessor(model_file=model_path)

    print("\n=== TOKENIZER INFO ===")
    print("Vocab size:", sp.get_piece_size())
    print("PAD:", sp.pad_id())
    print("UNK:", sp.unk_id())
    print("BOS:", sp.bos_id())
    print("EOS:", sp.eos_id())

    for text in [
        "I love machine translation.",
        "Tôi yêu thích dịch máy.",
        "Trí tuệ nhân tạo đang phát triển."
    ]:
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)

        print("\nOriginal:", text)
        print("Token IDs:", ids)
        print("Decoded:", decoded)

    assert sp.get_piece_size() == 8000
    assert sp.pad_id() == 0
    assert sp.unk_id() == 1
    assert sp.bos_id() == 2
    assert sp.eos_id() == 3


if __name__ == "__main__":
    prepare_corpus()
    train()
    verify()
