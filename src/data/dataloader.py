
from pathlib import Path
from functools import partial

import torch
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "shared_assets/data/iwslt15"

TOKENIZER_PATH = (
    ROOT / "shared_assets/tokenizer"
    / "spm_envi_bpe_8000.model"
)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
VOCAB_SIZE = 8000

def resolve_path(path):
    """Cho phép dùng đường dẫn tương đối từ thư mục dự án."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class IWSLT15Dataset(Dataset):
    def __init__(self, split, data_dir=None, tokenizer_path=None):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}")

        data_dir = resolve_path(
            data_dir if data_dir is not None else DATA_DIR
        )
        tokenizer_path = resolve_path(
            tokenizer_path if tokenizer_path is not None else TOKENIZER_PATH
        )

        if not tokenizer_path.is_file():
            raise FileNotFoundError(tokenizer_path)

        self.tokenizer = spm.SentencePieceProcessor(
            model_file=str(tokenizer_path)
        )

        assert self.tokenizer.get_piece_size() == VOCAB_SIZE
        assert self.tokenizer.pad_id() == PAD_ID
        assert self.tokenizer.unk_id() == UNK_ID
        assert self.tokenizer.bos_id() == BOS_ID
        assert self.tokenizer.eos_id() == EOS_ID

        src_path = data_dir / f"{split}.en"
        tgt_path = data_dir / f"{split}.vi"

        src_lines = src_path.read_text(
            encoding="utf-8"
        ).splitlines()

        tgt_lines = tgt_path.read_text(
            encoding="utf-8"
        ).splitlines()

        if len(src_lines) != len(tgt_lines):
            raise ValueError("Source/target count mismatch")

        if any(
            not src.strip() or not tgt.strip()
            for src, tgt in zip(src_lines, tgt_lines)
        ):
            raise ValueError("Empty sentence found")

        self.pairs = [
            (
                self.tokenizer.encode(src, out_type=int) + [EOS_ID],
                self.tokenizer.encode(tgt, out_type=int)
            )
            for src, tgt in zip(src_lines, tgt_lines)
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src, tgt = self.pairs[index]

        return {
            "src_ids": torch.tensor(src, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt, dtype=torch.long),
        }


def collate_fn(samples):
    src_list = []
    tgt_input_list = []
    tgt_target_list = []

    for sample in samples:
        src = sample["src_ids"]
        tgt = sample["tgt_ids"]

        tgt_input = torch.cat([
            torch.tensor([BOS_ID]),
            tgt
        ])

        tgt_target = torch.cat([
            tgt,
            torch.tensor([EOS_ID])
        ])

        src_list.append(src)
        tgt_input_list.append(tgt_input)
        tgt_target_list.append(tgt_target)

    src_ids = pad_sequence(
        src_list,
        batch_first=True,
        padding_value=PAD_ID
    )

    tgt_input_ids = pad_sequence(
        tgt_input_list,
        batch_first=True,
        padding_value=PAD_ID
    )

    tgt_target_ids = pad_sequence(
        tgt_target_list,
        batch_first=True,
        padding_value=PAD_ID
    )

    return {
        "src_ids": src_ids,
        "src_lengths": torch.tensor(
            [len(x) for x in src_list],
            dtype=torch.long
        ),
        "src_padding_mask": src_ids.eq(PAD_ID),
        "tgt_input_ids": tgt_input_ids,
        "tgt_target_ids": tgt_target_ids,
        "tgt_padding_mask": tgt_input_ids.eq(PAD_ID),
        "tgt_lengths": torch.tensor(
            [len(x) for x in tgt_target_list],
            dtype=torch.long
        )
    }


def get_dataloader(
    split,
    batch_size=32,
    shuffle=None,
    data_dir=None,
    tokenizer_path=None,
):
    dataset = IWSLT15Dataset(
        split,
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0
    )
