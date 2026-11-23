"""Public decoding contract: state and encoder context remain model-owned."""
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple
import torch
from torch import Tensor


class DecodingModel(Protocol):
    def encode(self, src_ids: Tensor, src_lengths: Tensor, src_padding_mask: Tensor) -> Any: ...
    def init_decoder_state(self, encoded: Any) -> Any: ...
    def decode_step(self, last_token: Tensor, state: Any) -> Tuple[Tensor, Any, Optional[Tensor]]: ...
    def reorder_state(self, state: Any, indices: Tensor) -> Any: ...


@dataclass(frozen=True)
class DecodeResult:
    token_ids: Tensor       # [B,L], excludes initial BOS, includes EOS, padded right
    lengths: Tensor         # [B], counts generated tokens including EOS, excludes PAD
    finished: Tensor        # [B] bool; True iff EOS generated
    log_scores: Tensor      # [B] accumulated log probability, before length penalty
    scores: Tensor          # [B] log_scores / length**alpha


def validate_inputs(model, src_ids, src_lengths, src_padding_mask, max_length, pad_id, bos_id, eos_id, alpha):
    if getattr(model, 'training', False):
        raise ValueError('call model.eval() before decoding; controller does not change model mode')
    if src_ids.ndim != 2 or src_ids.dtype != torch.long or src_ids.numel() == 0:
        raise ValueError('src_ids must be nonempty int64 [B,S]')
    if src_lengths.dtype != torch.long or src_lengths.shape != (src_ids.shape[0],):
        raise ValueError('src_lengths must be int64 [B]')
    if src_padding_mask.shape != src_ids.shape or src_padding_mask.dtype != torch.bool or src_padding_mask.device != src_ids.device:
        raise ValueError('src_padding_mask must be bool [B,S] on source device')
    lengths = src_lengths.to(src_ids.device)
    if bool(((lengths < 1) | (lengths > src_ids.shape[1])).any()):
        raise ValueError('invalid source lengths')
    if not torch.equal(src_padding_mask, torch.arange(src_ids.shape[1], device=src_ids.device)[None, :] >= lengths[:, None]):
        raise ValueError('mask must represent right padding and match lengths')
    if not torch.equal(src_ids.eq(pad_id), src_padding_mask):
        raise ValueError('source mask must match PAD tokens')
    if len({pad_id, bos_id, eos_id}) != 3 or min(pad_id, bos_id, eos_id) < 0:
        raise ValueError('PAD, BOS and EOS must be distinct nonnegative IDs')
    if max_length is not None and (not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1):
        raise ValueError('max_length must be a positive integer or None')
    if not isinstance(alpha, (int, float)) or not (0 <= alpha < float('inf')):
        raise ValueError('length_alpha must be finite and nonnegative')
    return [max_length if max_length is not None else int(n) + 50 for n in lengths.tolist()]


def log_probabilities(logits, batch_size, pad_id, bos_id, eos_id):
    if logits.ndim != 2 or logits.shape[0] != batch_size or not logits.is_floating_point():
        raise ValueError('decode_step logits must be floating [N,V]')
    if max(pad_id, bos_id, eos_id) >= logits.shape[1]:
        raise ValueError('special token ID outside vocabulary')
    if not torch.isfinite(logits).all():
        raise ValueError('decode_step logits must be finite')
    # Normalization over the full model vocabulary; then forbid generating PAD/BOS.
    probs = torch.log_softmax(logits.float(), dim=-1)
    probs[:, pad_id] = float('-inf')
    probs[:, bos_id] = float('-inf')
    return probs


def pack_results(records, device, pad_id):
    max_len = max(len(x[0]) for x in records)
    ids = torch.full((len(records), max_len), pad_id, dtype=torch.long, device=device)
    for i, (seq, _, _, _) in enumerate(records):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return DecodeResult(ids,
                        torch.tensor([len(x[0]) for x in records], dtype=torch.long, device=device),
                        torch.tensor([x[1] for x in records], dtype=torch.bool, device=device),
                        torch.tensor([x[2] for x in records], dtype=torch.float64, device=device),
                        torch.tensor([x[3] for x in records], dtype=torch.float64, device=device))
