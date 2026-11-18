from pathlib import Path

import sentencepiece as spm


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "shared_assets/tokenizer/spm_envi_bpe_8000.model"
)


def test_shared_tokenizer():
    assert MODEL_PATH.is_file()
    tokenizer = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))

    vocab_size = tokenizer.get_piece_size()
    assert vocab_size == 8000
    assert tokenizer.pad_id() == 0
    assert tokenizer.unk_id() == 1
    assert tokenizer.bos_id() == 2
    assert tokenizer.eos_id() == 3

    for sample in (
        'She said, "Hello!"',
        'Cô ấy nói: "Xin chào."',
        "Tôi yêu thích dịch máy.",
    ):
        token_ids = tokenizer.encode(sample, out_type=int)
        assert token_ids, sample
        assert all(0 <= token_id < vocab_size for token_id in token_ids), sample
        assert tokenizer.decode(token_ids) == sample


if __name__ == "__main__":
    test_shared_tokenizer()
    print("ALL TOKENIZER TESTS PASSED")
