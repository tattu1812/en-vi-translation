
import torch
from src.data.dataloader import (
    get_dataloader, PAD_ID, BOS_ID, EOS_ID
)


def main():
    expected = {
        "train": 131189,
        "validation": 1553,
        "test": 1268
    }

    for split, count in expected.items():
        loader = get_dataloader(
            split,
            batch_size=4,
            shuffle=False
        )

        assert len(loader.dataset) == count

        batch = next(iter(loader))

        src = batch["src_ids"]
        tgt_in = batch["tgt_input_ids"]
        tgt_out = batch["tgt_target_ids"]

        assert src.shape[0] == 4
        assert tgt_in.shape == tgt_out.shape

        assert torch.equal(
            batch["src_padding_mask"],
            src.eq(PAD_ID)
        )

        assert torch.equal(
            batch["tgt_padding_mask"],
            tgt_in.eq(PAD_ID)
        )

        for i in range(4):
            src_len = batch["src_lengths"][i].item()
            tgt_len = batch["tgt_lengths"][i].item()

            assert src[i, src_len - 1] == EOS_ID
            assert tgt_in[i, 0] == BOS_ID
            assert tgt_out[i, tgt_len - 1] == EOS_ID

            assert torch.equal(
                tgt_in[i, 1:tgt_len],
                tgt_out[i, :tgt_len - 1]
            )

            assert torch.all(
                tgt_out[i, tgt_len:] == PAD_ID
            )

        print(
            f"PASS {split}: {count} pairs, "
            f"src={tuple(src.shape)}, "
            f"tgt={tuple(tgt_in.shape)}"
        )

    print("\nALL DATALOADER TESTS PASSED")


if __name__ == "__main__":
    main()
