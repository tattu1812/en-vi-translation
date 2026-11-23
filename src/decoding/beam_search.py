"""Beam search with a separate bounded set of completed hypotheses."""
import torch
from .greedy import greedy_decode
from .interface import validate_inputs, log_probabilities, pack_results


@torch.inference_mode()
def beam_search(model, src_ids, src_lengths, src_padding_mask, *, beam_size=5, max_length=None,
                pad_id=0, bos_id=2, eos_id=3, length_alpha=1.0):
    limits = validate_inputs(model, src_ids, src_lengths, src_padding_mask, max_length,
                             pad_id, bos_id, eos_id, length_alpha)
    if not isinstance(beam_size, int) or isinstance(beam_size, bool) or beam_size < 1:
        raise ValueError('beam_size must be a positive integer')
    # Width 1 is defined to be exactly greedy, including score and tie semantics.
    if beam_size == 1:
        return greedy_decode(model, src_ids, src_lengths, src_padding_mask, max_length=max_length,
                             pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, length_alpha=length_alpha)
    device = src_ids.device
    initial = model.init_decoder_state(model.encode(src_ids, src_lengths, src_padding_mask))
    results = []
    # Isolated per-source beams: no beam may select a parent from another source.
    for sample, limit in enumerate(limits):
        state = model.reorder_state(initial, torch.tensor([sample], dtype=torch.long, device=device))
        live_sequences, live_scores = [[]], [0.0]
        tokens = torch.tensor([bos_id], dtype=torch.long, device=device)
        completed = []
        for step in range(1, limit + 1):
            logits, updated, _ = model.decode_step(tokens, state)
            if logits.device != device:
                raise ValueError('decode_step logits must be on the source device')
            probs = log_probabilities(logits, len(live_sequences), pad_id, bos_id, eos_id)
            # Save EOS separately, so a later live beam cannot overwrite a finished one.
            for parent, seq in enumerate(live_sequences):
                raw = live_scores[parent] + float(probs[parent, eos_id].item())
                completed.append((seq + [eos_id], True, raw, raw / (step ** length_alpha)))
            completed.sort(key=lambda h: (-h[3], tuple(h[0])))
            completed = completed[:beam_size]
            # All live candidates have the same length, so raw-score top-k is also
            # normalized-score top-k. Stable sort: ties by parent then token ID.
            scores = probs.to(torch.float64) + torch.tensor(live_scores, dtype=torch.float64, device=device)[:, None]
            scores[:, eos_id] = float('-inf')
            flat = scores.flatten()
            order = torch.argsort(flat, descending=True, stable=True)
            chosen = order[torch.isfinite(flat[order])][:beam_size]
            if chosen.numel() == 0:
                live_sequences, live_scores = [], []
                break
            vocab = probs.shape[1]
            parents = torch.div(chosen, vocab, rounding_mode='floor')
            selected = chosen.remainder(vocab)
            old_sequences = live_sequences
            live_sequences = [old_sequences[p] + [t] for p, t in zip(parents.tolist(), selected.tolist())]
            live_scores = flat[chosen].tolist()
            state = model.reorder_state(updated, parents)
            tokens = selected
        unfinished = [(seq, False, raw, raw / (len(seq) ** length_alpha))
                      for seq, raw in zip(live_sequences, live_scores) if seq]
        candidates = completed + unfinished
        if not candidates:
            raise RuntimeError('no valid decoding hypothesis')
        # Final rank applies once to completed and capped unfinished hypotheses.
        candidates.sort(key=lambda h: (-h[3], tuple(h[0])))
        results.append(candidates[0])
    return pack_results(results, device, pad_id)
