
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.data.dataloader import (
    get_dataloader,
    TOKENIZER_PATH,
    PAD_ID,
    BOS_ID,
    EOS_ID,
)


def check_split(split, count):
    loader = get_dataloader(split, batch_size=4, shuffle=False)

    assert len(loader.dataset) == count

    batch = next(iter(loader))

    src = batch["src_ids"]
    tgt_in = batch["tgt_input_ids"]
    tgt_out = batch["tgt_target_ids"]

    assert src.shape[0] == 4
    assert tgt_in.shape == tgt_out.shape

    assert torch.equal(
        batch["src_padding_mask"],
        src.eq(PAD_ID),
    )

    assert torch.equal(
        batch["tgt_padding_mask"],
        tgt_in.eq(PAD_ID),
    )

    for i in range(4):
        src_len = batch["src_lengths"][i].item()
        tgt_len = batch["tgt_lengths"][i].item()

        assert src[i, src_len - 1].item() == EOS_ID
        assert tgt_in[i, 0].item() == BOS_ID
        assert tgt_out[i, tgt_len - 1].item() == EOS_ID

        assert torch.equal(
            tgt_in[i, 1:tgt_len],
            tgt_out[i, :tgt_len - 1],
        )

        assert torch.all(
            tgt_out[i, tgt_len:] == PAD_ID
        )

    print(
        f"PASS {split}: {count} pairs, "
        f"src={tuple(src.shape)}, "
        f"tgt={tuple(tgt_in.shape)}"
    )


def check_invalid_inputs():
    with TemporaryDirectory() as directory:
        data_dir = Path(directory)
        en_path = data_dir / "test.en"
        vi_path = data_dir / "test.vi"
        en_path.write_text("Hello.\nGood morning.\n", encoding="utf-8")

        for vi_text, message in (
            ("Xin chào.\n", "Source/target count mismatch"),
            ("Xin chào.\n\n", "Empty sentence found"),
        ):
            vi_path.write_text(vi_text, encoding="utf-8")
            try:
                get_dataloader("test", data_dir=data_dir, tokenizer_path=TOKENIZER_PATH)
            except ValueError as error:
                assert str(error) == message
            else:
                raise AssertionError(f"Expected ValueError: {message}")

        missing_path = data_dir / "missing.model"
        try:
            get_dataloader("test", data_dir=data_dir, tokenizer_path=missing_path)
        except FileNotFoundError as error:
            assert error.args == (missing_path,)
        else:
            raise AssertionError("Expected FileNotFoundError for missing tokenizer")

    print("PASS invalid inputs: line count mismatch, empty sentence, missing tokenizer")


def main():
    print("\n=== DATALOADER V3 ===")
    for split, count in (("train", 131000), ("validation", 1553), ("test", 1268)):
        check_split(split, count)

    check_invalid_inputs()
    print("\nALL DATALOADER TESTS PASSED")


if __name__ == "__main__":
    main()
