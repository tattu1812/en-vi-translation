"""Proposal-aligned BiLSTM / Luong-general model, without a training loop.

D02's common inference interface is provisional until agreed with Duc. The
opaque DecoderState owns its encoder memory, mask, recurrent state and the
previous attentional vector, so a decoding controller need only reorder it.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

VOCAB_SIZE = 8000
PAD_ID = 0
EMBEDDING_DIM = 512
HIDDEN_DIM = 512
ENCODER_DIM = 1024
NUM_LAYERS = 2


@dataclass(frozen=True)
class EncoderOutput:
    memory: Tensor
    mask: Tensor
    hidden: Tensor
    cell: Tensor


@dataclass(frozen=True)
class DecoderState:
    hidden: Tensor
    cell: Tensor
    input_feed: Tensor
    memory: Tensor
    mask: Tensor


class Encoder(nn.Module):
    """Two-layer BiLSTM; the caller supplies the shared embedding."""

    def __init__(self, embedding: nn.Embedding, dropout: float = 0.3):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
                            batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, src_ids: Tensor, src_lengths: Tensor):
        if src_ids.ndim != 2 or src_ids.dtype != torch.long or src_ids.numel() == 0:
            raise ValueError("src_ids must be a nonempty [B,S] int64 tensor")
        batch, steps = src_ids.shape
        if src_lengths.shape != (batch,) or src_lengths.dtype != torch.long:
            raise ValueError("src_lengths must be int64 [B]")
        lengths = src_lengths.detach().to(device="cpu")
        if bool(((lengths < 1) | (lengths > steps)).any()):
            raise ValueError("source lengths must be in [1,S]")
        packed = pack_padded_sequence(self.dropout(self.embedding(src_ids)), lengths,
                                      batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        memory, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=steps)
        return memory, hidden, cell


class Bridge(nn.Module):
    """D01: concatenate forward/backward per layer and project h and c separately."""

    def __init__(self):
        super().__init__()
        self.hidden_projection = nn.ModuleList(nn.Linear(ENCODER_DIM, HIDDEN_DIM)
                                                for _ in range(NUM_LAYERS))
        self.cell_projection = nn.ModuleList(nn.Linear(ENCODER_DIM, HIDDEN_DIM)
                                              for _ in range(NUM_LAYERS))

    def forward(self, hidden: Tensor, cell: Tensor):
        if (hidden.ndim != 3 or hidden.shape[0] != 2 * NUM_LAYERS
                or hidden.shape[2] != HIDDEN_DIM or cell.shape != hidden.shape):
            raise ValueError("encoder hidden and cell must be [4,B,512]")
        h_out, c_out = [], []
        for layer in range(NUM_LAYERS):
            h_pair = torch.cat((hidden[2 * layer], hidden[2 * layer + 1]), dim=-1)
            c_pair = torch.cat((cell[2 * layer], cell[2 * layer + 1]), dim=-1)
            h_out.append(torch.tanh(self.hidden_projection[layer](h_pair)))
            c_out.append(torch.tanh(self.cell_projection[layer](c_pair)))
        return torch.stack(h_out), torch.stack(c_out)


class LuongGeneralAttention(nn.Module):
    """score(q,H_s) = q.T W H_s; mask convention is True = PAD."""

    def __init__(self):
        super().__init__()
        self.key_projection = nn.Linear(ENCODER_DIM, HIDDEN_DIM, bias=False)

    def forward(self, query: Tensor, memory: Tensor, mask: Tensor):
        if (query.ndim != 2 or query.shape[-1] != HIDDEN_DIM or memory.ndim != 3
                or memory.shape[0] != query.shape[0] or memory.shape[-1] != ENCODER_DIM
                or mask.shape != memory.shape[:2] or mask.dtype != torch.bool):
            raise ValueError("expected query [B,512], memory [B,S,1024], bool mask [B,S]")
        if memory.shape[1] == 0 or bool(mask.all(dim=1).any()):
            raise ValueError("each source must have at least one non-PAD token")
        scores = torch.bmm(self.key_projection(memory), query.unsqueeze(-1)).squeeze(-1)
        # Accumulate softmax in FP32 for half precision, retain float64 if supplied.
        softmax_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
        weights = torch.softmax(scores.to(softmax_dtype).masked_fill(mask, float('-inf')), dim=-1)
        context = torch.bmm(weights.to(memory.dtype).unsqueeze(1), memory).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    """Luong input feeding: feed previous attentional vector, not raw context."""

    def __init__(self, embedding: nn.Embedding, dropout: float = 0.3):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(EMBEDDING_DIM + HIDDEN_DIM, HIDDEN_DIM, NUM_LAYERS,
                            batch_first=True, dropout=dropout)
        self.attention = LuongGeneralAttention()
        self.combine = nn.Linear(HIDDEN_DIM + ENCODER_DIM, HIDDEN_DIM)
        self.output_projection = nn.Linear(EMBEDDING_DIM, embedding.num_embeddings)
        # Tie the actual Parameter; this is separate from encoder/decoder sharing.
        self.output_projection.weight = self.embedding.weight

    def forward(self, token: Tensor, state: DecoderState):
        batch = state.memory.shape[0]
        if token.shape != (batch,) or token.dtype != torch.long:
            raise ValueError("decoder token must be int64 [B]")
        if state.input_feed.shape != (batch, HIDDEN_DIM):
            raise ValueError("input feeding state must be [B,512]")
        embedded = self.dropout(self.embedding(token))
        decoder_input = torch.cat((embedded, state.input_feed), dim=-1).unsqueeze(1)
        output, (hidden, cell) = self.lstm(decoder_input, (state.hidden, state.cell))
        query = output[:, 0, :]
        context, weights = self.attention(query, state.memory, state.mask)
        attentional = torch.tanh(self.combine(torch.cat((query, context), dim=-1)))
        logits = self.output_projection(self.dropout(attentional))
        new_state = DecoderState(hidden, cell, attentional, state.memory, state.mask)
        return logits, new_state, weights


class Seq2SeqLSTM(nn.Module):
    """Shared EN/VI embedding, tied output, 100% teacher forcing in forward."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=PAD_ID)
        self.encoder = Encoder(self.embedding, dropout)
        self.bridge = Bridge()
        self.decoder = Decoder(self.embedding, dropout)

    def encode(self, src_ids: Tensor, src_lengths: Tensor, src_padding_mask: Tensor):
        if (src_padding_mask.dtype != torch.bool or src_padding_mask.shape != src_ids.shape
                or src_padding_mask.device != src_ids.device):
            raise ValueError("source padding mask must be bool [B,S] on the source device")
        if not torch.equal(src_padding_mask, src_ids.eq(PAD_ID)):
            raise ValueError("source padding mask must mark PAD IDs")
        if src_ids.ndim != 2 or src_lengths.shape != (src_ids.shape[0],):
            raise ValueError("source IDs and lengths must be [B,S] and [B]")
        expected_mask = torch.arange(src_ids.shape[1], device=src_ids.device).unsqueeze(0) >= src_lengths.to(src_ids.device).unsqueeze(1)
        if not torch.equal(src_padding_mask, expected_mask):
            raise ValueError("source must be right-padded and lengths must match the mask")
        memory, hidden, cell = self.encoder(src_ids, src_lengths)
        return EncoderOutput(memory, src_padding_mask, hidden, cell)

    def init_decoder_state(self, encoded: EncoderOutput):
        hidden, cell = self.bridge(encoded.hidden, encoded.cell)
        feed = encoded.memory.new_zeros((encoded.memory.shape[0], HIDDEN_DIM))
        return DecoderState(hidden, cell, feed, encoded.memory, encoded.mask)

    def decode_step(self, last_token: Tensor, state: DecoderState):
        return self.decoder(last_token, state)

    def reorder_state(self, state: DecoderState, indices: Tensor):
        if indices.ndim != 1 or indices.dtype != torch.long or indices.device != state.memory.device:
            raise ValueError("indices must be int64 [N] on the state device")
        if bool(((indices < 0) | (indices >= state.memory.shape[0])).any()):
            raise ValueError("reorder index out of bounds")
        return DecoderState(state.hidden.index_select(1, indices),
                            state.cell.index_select(1, indices),
                            state.input_feed.index_select(0, indices),
                            state.memory.index_select(0, indices),
                            state.mask.index_select(0, indices))

    def forward(self, src_ids: Tensor, src_lengths: Tensor, src_padding_mask: Tensor,
                tgt_input_ids: Tensor, return_attention: bool = False):
        if (tgt_input_ids.ndim != 2 or tgt_input_ids.dtype != torch.long
                or tgt_input_ids.shape[0] != src_ids.shape[0] or tgt_input_ids.shape[1] == 0):
            raise ValueError("target input must be nonempty int64 [B,T]")
        state = self.init_decoder_state(self.encode(src_ids, src_lengths, src_padding_mask))
        logits, attentions = [], []
        for token in tgt_input_ids.unbind(dim=1):
            step_logits, state, weights = self.decode_step(token, state)
            logits.append(step_logits)
            if return_attention:
                attentions.append(weights)
        result = torch.stack(logits, dim=1)
        return (result, torch.stack(attentions, dim=1)) if return_attention else result


def translation_loss(logits: Tensor, tgt_target_ids: Tensor, label_smoothing: float = 0.1):
    """Cross-entropy over non-PAD labels including EOS; no softmax before loss."""
    if (logits.ndim != 3 or tgt_target_ids.shape != logits.shape[:2]
            or tgt_target_ids.dtype != torch.long):
        raise ValueError("logits and targets must be [B,T,V] and int64 [B,T]")
    if not bool(tgt_target_ids.ne(PAD_ID).any()):
        raise ValueError("loss requires at least one non-PAD target")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_target_ids.reshape(-1),
                           ignore_index=PAD_ID, label_smoothing=label_smoothing)
