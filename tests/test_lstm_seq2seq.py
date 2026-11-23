"""Focused unit and V3 integration tests; no training or checkpoint files."""
import io
import math
import time
import unittest

import torch
from torch.nn import functional as F

from src.data.dataloader import get_dataloader, TOKENIZER_PATH
from src.models.lstm_seq2seq import (
    Bridge, DecoderState, LuongGeneralAttention, Seq2SeqLSTM, translation_loss,
)

# Avoid excessive CPU thread overhead on very small batches.
torch.set_num_threads(2)


class TestLSTMSeq2Seq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.model = Seq2SeqLSTM().eval()
        cls.src = torch.tensor([[5, 6, 3, 0], [7, 3, 0, 0], [8, 9, 10, 3]])
        cls.lengths = torch.tensor([3, 2, 4])  # deliberately unsorted
        cls.mask = cls.src.eq(0)

    def test_01_encoder_shapes_order_and_padding(self):
        with torch.no_grad():
            memory, hidden, cell = self.model.encoder(self.src, self.lengths)
            self.assertEqual(memory.shape, (3, 4, 1024))
            self.assertEqual(hidden.shape, (4, 3, 512))
            self.assertEqual(cell.shape, (4, 3, 512))
            for i, length in enumerate(self.lengths.tolist()):
                one, h, c = self.model.encoder(self.src[i:i+1, :length], self.lengths[i:i+1])
                torch.testing.assert_close(memory[i, :length], one[0], rtol=1e-4, atol=1e-6)
                torch.testing.assert_close(hidden[:, i:i+1], h, rtol=1e-4, atol=1e-6)
                torch.testing.assert_close(cell[:, i:i+1], c, rtol=1e-4, atol=1e-6)
                self.assertTrue(torch.count_nonzero(memory[i, length:]) == 0)

    def test_02_bridge_directions_and_gradient(self):
        bridge = Bridge()
        with torch.no_grad():
            for layer in range(2):
                for proj in (bridge.hidden_projection[layer], bridge.cell_projection[layer]):
                    proj.weight.zero_(); proj.bias.zero_()
                    proj.weight[:, :512] = torch.eye(512)
                    proj.weight[:, 512:] = 2 * torch.eye(512)
        hidden = torch.stack([torch.full((1, 512), x) for x in (0.1, 0.2, 0.3, 0.4)]).requires_grad_()
        cell = (hidden.detach() * 0.5).requires_grad_()
        h, c = bridge(hidden, cell)
        self.assertEqual(h.shape, (2, 1, 512))
        self.assertEqual(c.shape, (2, 1, 512))
        torch.testing.assert_close(h[:, 0, 0], torch.tanh(torch.tensor([0.5, 1.1])))
        torch.testing.assert_close(c[:, 0, 0], torch.tanh(torch.tensor([0.25, 0.55])))
        (h.sum() + c.sum()).backward()
        self.assertTrue(torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0)
        self.assertTrue(torch.isfinite(cell.grad).all() and cell.grad.abs().sum() > 0)
        self.assertIsNot(bridge.hidden_projection[0].weight, bridge.cell_projection[0].weight)

    def test_03_attention_formula_mask_and_all_pad(self):
        torch.manual_seed(3)
        attention = LuongGeneralAttention().eval()
        q, mem = torch.randn(2, 512), torch.randn(2, 4, 1024)
        mask = torch.tensor([[False, False, True, True], [False, False, False, True]])
        context, weights = attention(q, mem, mask)
        scores = torch.einsum('bd,de,bse->bs', q, attention.key_projection.weight, mem)
        expected = scores.masked_fill(mask, float('-inf')).softmax(-1)
        torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(context, torch.einsum('bs,bse->be', expected, mem))
        self.assertEqual(context.shape, (2, 1024))
        self.assertTrue(torch.equal(weights[mask], torch.zeros_like(weights[mask])))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        self.assertTrue(torch.isfinite(context).all() and torch.isfinite(weights).all())
        altered = mem.clone(); altered[mask] = 1000
        torch.testing.assert_close(attention(q, altered, mask)[0], context)
        with self.assertRaisesRegex(ValueError, 'non-PAD'):
            attention(q, mem, torch.ones_like(mask))

    def test_04_decoder_step_sharing_and_input_feed(self):
        self.assertIs(self.model.encoder.embedding.weight, self.model.decoder.embedding.weight)
        self.assertIs(self.model.decoder.output_projection.weight, self.model.decoder.embedding.weight)
        with torch.no_grad():
            state = self.model.init_decoder_state(self.model.encode(self.src[:2], self.lengths[:2], self.mask[:2]))
            tok = torch.tensor([2, 2])
            logits, updated, attention = self.model.decode_step(tok, state)
            changed = DecoderState(state.hidden, state.cell, torch.ones_like(state.input_feed), state.memory, state.mask)
            logits_changed, _, _ = self.model.decode_step(tok, changed)
        self.assertEqual(logits.shape, (2, 8000))
        self.assertEqual(attention.shape, (2, 4))
        self.assertEqual(updated.hidden.shape, (2, 2, 512))
        self.assertFalse(torch.allclose(logits, logits_changed))
        self.assertFalse(torch.allclose(updated.hidden, state.hidden))
        self.assertFalse(torch.allclose(updated.input_feed, state.input_feed))
        self.assertEqual(torch.count_nonzero(state.input_feed).item(), 0)

    def test_05_forward_equals_step_loop_and_batch_one(self):
        target = torch.tensor([[2, 41, 42], [2, 17, 0]])
        with torch.no_grad():
            logits, attention = self.model(self.src[:2], self.lengths[:2], self.mask[:2], target, True)
            state = self.model.init_decoder_state(self.model.encode(self.src[:2], self.lengths[:2], self.mask[:2]))
            steps=[]
            for token in target.unbind(1):
                step, state, _ = self.model.decode_step(token, state); steps.append(step)
            torch.testing.assert_close(logits, torch.stack(steps, 1))
            one = self.model(self.src[:1], self.lengths[:1], self.mask[:1], target[:1])
        self.assertEqual(logits.shape, (2, 3, 8000))
        self.assertEqual(attention.shape, (2, 3, 4))
        self.assertEqual(one.shape, (1, 3, 8000))

    def test_06_reorder_duplicate_parent_and_validation(self):
        with torch.no_grad():
            state = self.model.init_decoder_state(self.model.encode(self.src[:2], self.lengths[:2], self.mask[:2]))
            _, state, _ = self.model.decode_step(torch.tensor([2, 2]), state)
            ids = torch.tensor([1, 1, 0]); reordered = self.model.reorder_state(state, ids)
            for name, axis in [('hidden', 1), ('cell', 1), ('input_feed', 0), ('memory', 0), ('mask', 0)]:
                self.assertTrue(torch.equal(getattr(reordered, name), getattr(state, name).index_select(axis, ids)))
            with self.assertRaises(ValueError): self.model.reorder_state(state, torch.tensor([2]))
            with self.assertRaises(ValueError): self.model.encode(self.src, self.lengths, ~self.mask)

    def test_07_loss_ignore_pad_and_eos(self):
        torch.manual_seed(8)
        logits = torch.randn(1, 2, 8000, requires_grad=True)
        targets = torch.tensor([[3, 0]])
        loss = translation_loss(logits, targets)
        expected = F.cross_entropy(logits[:, :1, :].reshape(-1, 8000), torch.tensor([3]), label_smoothing=.1)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.count_nonzero(logits.grad[:, 0]) > 0)
        self.assertEqual(torch.count_nonzero(logits.grad[:, 1]).item(), 0)
        with self.assertRaises(ValueError): translation_loss(logits.detach(), torch.zeros_like(targets))

    def test_08_save_load_in_memory(self):
        with torch.no_grad():
            target = torch.tensor([[2, 4]])
            before = self.model(self.src[:1], self.lengths[:1], self.mask[:1], target)
            buffer = io.BytesIO(); torch.save(self.model.state_dict(), buffer); buffer.seek(0)
            restored = Seq2SeqLSTM().eval()
            restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True))
            after = restored(self.src[:1], self.lengths[:1], self.mask[:1], target)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertIs(restored.encoder.embedding.weight, restored.decoder.output_projection.weight)

    def test_09_dataloader_v3_forward_backward(self):
        import sentencepiece as spm
        import sys
        sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
        self.assertEqual((sp.get_piece_size(), sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()), (8000, 0, 1, 2, 3))
        # Actual first sample has source 14 and target 11 BPE tokens; no truncation.
        batch = next(iter(get_dataloader('train', batch_size=1, shuffle=False)))
        self.assertTrue(torch.equal(batch['src_padding_mask'], batch['src_ids'].eq(0)))
        n = batch['tgt_lengths'][0].item()
        self.assertEqual(batch['tgt_input_ids'][0, 0].item(), 2)
        self.assertEqual(batch['tgt_target_ids'][0, n-1].item(), 3)
        self.assertTrue(torch.equal(batch['tgt_input_ids'][0, 1:n], batch['tgt_target_ids'][0, :n-1]))
        self.model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        logits, attention = self.model(batch['src_ids'], batch['src_lengths'], batch['src_padding_mask'], batch['tgt_input_ids'], True)
        loss = translation_loss(logits, batch['tgt_target_ids'])
        loss.backward()
        elapsed = time.perf_counter() - start
        self.assertEqual(logits.shape, (1, n, 8000))
        self.assertEqual(attention.shape, (1, n, batch['src_ids'].shape[1]))
        self.assertTrue(math.isfinite(loss.item()))
        grads = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(g.abs().sum().item() > 0 for g in grads))
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        print(f'\nINTEGRATION PASS python={sys.version.split()[0]} torch={torch.__version__} sentencepiece={spm.__version__} '
              f'src={tuple(batch["src_ids"].shape)} target={tuple(batch["tgt_input_ids"].shape)} '
              f'logits={tuple(logits.shape)} attention={tuple(attention.shape)} loss={loss.item():.6f} '
              f'params={total} trainable={trainable} parameter_bytes={param_bytes} forward_backward_seconds={elapsed:.3f}', flush=True)
        self.model.zero_grad(set_to_none=True)


    def test_10_train_mode_dropout_backward(self):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        try:
            targets = torch.tensor([[2, 14], [2, 9]])
            labels = torch.tensor([[14, 3], [9, 3]])
            logits = self.model(self.src[:2], self.lengths[:2], self.mask[:2], targets)
            loss = translation_loss(logits, labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
            self.assertIsNotNone(self.model.decoder.lstm.weight_ih_l0.grad)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.eval()


if __name__ == '__main__':
    unittest.main(verbosity=2)
