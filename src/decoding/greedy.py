"""Greedy decoding through an opaque model-owned state."""
import torch
from .interface import validate_inputs, log_probabilities, pack_results


@torch.inference_mode()
def greedy_decode(model, src_ids, src_lengths, src_padding_mask, *, max_length=None,
                  pad_id=0, bos_id=2, eos_id=3, length_alpha=1.0):
    limits = validate_inputs(model, src_ids, src_lengths, src_padding_mask, max_length,
                             pad_id, bos_id, eos_id, length_alpha)
    device = src_ids.device
    state = model.init_decoder_state(model.encode(src_ids, src_lengths, src_padding_mask))
    active = list(range(src_ids.shape[0]))
    tokens = torch.full((len(active),), bos_id, dtype=torch.long, device=device)
    sequences = [[] for _ in active]
    totals = [0.0 for _ in active]
    ended = [False for _ in active]
    while active:
        logits, updated, _ = model.decode_step(tokens, state)
        if logits.device != device:
            raise ValueError('decode_step logits must be on the source device')
        probs = log_probabilities(logits, len(active), pad_id, bos_id, eos_id)
        # torch.argmax chooses the lowest token ID on exact ties.
        selected = probs.argmax(dim=-1)
        keep, next_active = [], []
        for i, sample in enumerate(active):
            token = int(selected[i].item())
            sequences[sample].append(token)
            totals[sample] += float(probs[i, token].item())
            ended[sample] = token == eos_id
            if not ended[sample] and len(sequences[sample]) < limits[sample]:
                keep.append(i)
                next_active.append(sample)
        if not keep:
            break
        indices = torch.tensor(keep, dtype=torch.long, device=device)
        state = model.reorder_state(updated, indices)
        tokens = selected.index_select(0, indices)
        active = next_active
    return pack_results([(seq, done, total, total / (len(seq) ** length_alpha))
                         for seq, done, total in zip(sequences, ended, totals)], device, pad_id)
