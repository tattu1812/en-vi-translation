"""Architecture-independent decoding contract and search regression tests."""
import itertools
import math
import unittest
from dataclasses import dataclass

import torch
from src.decoding import beam_search, greedy_decode
from src.models.lstm_seq2seq import Seq2SeqLSTM

torch.set_num_threads(2)


@dataclass(frozen=True)
class MockState:
    source: torch.Tensor
    prefix: torch.Tensor


class MockTransformer:
    """A fake prefix-based adapter, NOT a real Transformer implementation."""
    training = False
    def __init__(self, mode='normal'):
        self.mode = mode
        self.reorders = []
        self.calls = []

    def encode(self, src_ids, src_lengths, src_padding_mask):
        return src_ids[:, 0].clone()

    def init_decoder_state(self, encoded):
        return MockState(encoded, torch.empty((len(encoded), 0), dtype=torch.long, device=encoded.device))

    def decode_step(self, last_token, state):
        self.calls.append((last_token.clone(), state.source.clone(), state.prefix.clone()))
        # The first call consumes BOS but does not append it to prefix.
        prefix = state.prefix if state.prefix.shape[1] == 0 and bool((last_token == 2).all()) else torch.cat((state.prefix, last_token[:, None]), 1)
        logits = torch.full((len(last_token), 6), -8.0, device=last_token.device)
        for i, source in enumerate(state.source.tolist()):
            history = tuple(prefix[i].tolist())
            if self.mode == 'never':
                logits[i, 4] = 4; logits[i, 5] = 2; logits[i, 3] = -20
            elif self.mode == 'ties':
                logits[i, 4] = logits[i, 5] = 2; logits[i, 3] = -20
            elif self.mode == 'eos_first':
                logits[i, 3] = 5; logits[i, 4] = 2; logits[i, 5] = 1
            elif not history:
                logits[i, 4] = 3 if source == 7 else 1
                logits[i, 5] = 1 if source == 7 else 3
                logits[i, 3] = -0.5
            elif len(history) == 1:
                logits[i, 3] = 4 if (history[0] == 4) == (source == 7) else -1
                logits[i, 4] = 1; logits[i, 5] = 2
            else:
                logits[i, 3] = 4; logits[i, 4] = 1; logits[i, 5] = 0
        return logits, MockState(state.source, prefix), None

    def reorder_state(self, state, indices):
        self.reorders.append(indices.tolist())
        return MockState(state.source.index_select(0, indices), state.prefix.index_select(0, indices))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.src = torch.tensor([[7, 3, 0], [8, 3, 0]])
        self.lengths = torch.tensor([2, 2])
        self.mask = self.src.eq(0)

    def test_greedy_batch_argmax_eos_no_mutation(self):
        model=MockTransformer()
        out=greedy_decode(model,self.src,self.lengths,self.mask,max_length=5)
        self.assertEqual(out.token_ids.tolist(), [[4,3],[5,3]])
        self.assertEqual(out.lengths.tolist(), [2,2])
        self.assertEqual(out.finished.tolist(), [True,True])
        self.assertEqual(model.calls[0][0].tolist(), [2,2])
        self.assertFalse(out.token_ids.requires_grad)

    def test_greedy_ends_and_removes_completed(self):
        model=MockTransformer()
        # A custom model makes the first source finish immediately.
        original=model.decode_step
        def step(token,state):
            logits, state, attn=original(token,state)
            if state.prefix.shape[1]==0: logits[state.source==7,3]=10
            return logits,state,attn
        model.decode_step=step
        out=greedy_decode(model,self.src,self.lengths,self.mask,max_length=4)
        self.assertEqual(out.token_ids.tolist(), [[3,0],[5,3]])
        self.assertEqual(out.lengths.tolist(),[1,2])
        self.assertEqual(model.calls[1][1].tolist(),[8])

    def test_greedy_max_length_and_pad_bos_forbidden(self):
        model=MockTransformer('never')
        out=greedy_decode(model,self.src[:1],self.lengths[:1],self.mask[:1],max_length=3)
        self.assertEqual(out.token_ids.tolist(),[[4,4,4]])
        self.assertEqual(out.finished.tolist(),[False])
        self.assertEqual(out.lengths.tolist(),[3])

    def test_exact_tie_lowest_token_and_beam_one(self):
        for mode in ('ties','normal','eos_first'):
            g=greedy_decode(MockTransformer(mode),self.src,self.lengths,self.mask,max_length=3)
            b=beam_search(MockTransformer(mode),self.src,self.lengths,self.mask,beam_size=1,max_length=3)
            for field in ('token_ids','lengths','finished','log_scores','scores'):
                self.assertTrue(torch.equal(getattr(g,field),getattr(b,field)),field)
        self.assertEqual(greedy_decode(MockTransformer('ties'),self.src[:1],self.lengths[:1],self.mask[:1],max_length=2).token_ids.tolist(),[[4,4]])

    def test_beam_matches_exhaustive_enumeration(self):
        # All possible content prefixes are retained with K=5, horizon=3.
        model=MockTransformer()
        out=beam_search(model,self.src[:1],self.lengths[:1],self.mask[:1],beam_size=5,max_length=3,length_alpha=0.7)
        candidates=[]
        for sequence in [(x,) for x in (3,4,5)]+list(itertools.product((4,5),(3,4,5)))+list(itertools.product((4,5),(4,5),(3,4,5))):
            if sequence[-1] != 3 and len(sequence)<3: continue
            oracle=MockTransformer(); state=oracle.init_decoder_state(torch.tensor([7])); prev=torch.tensor([2]); raw=0
            for token in sequence:
                logits,state,_=oracle.decode_step(prev,state)
                raw+=float(logits.log_softmax(-1)[0,token]); prev=torch.tensor([token])
            candidates.append((raw/(len(sequence)**0.7), sequence, raw))
        candidates.sort(key=lambda item:(-item[0],item[1]))
        score, seq, raw=candidates[0]
        self.assertEqual(tuple(out.token_ids[0,:out.lengths[0]].tolist()),seq)
        self.assertAlmostEqual(out.log_scores.item(),raw,places=6)
        self.assertAlmostEqual(out.scores.item(),score,places=6)

    def test_beam_batch_isolation_and_reorder(self):
        m=MockTransformer()
        out=beam_search(m,self.src,self.lengths,self.mask,beam_size=5,max_length=3)
        for i in range(2):
            separate=beam_search(MockTransformer(),self.src[i:i+1],self.lengths[i:i+1],self.mask[i:i+1],beam_size=5,max_length=3)
            self.assertEqual(out.token_ids[i,:out.lengths[i]].tolist(),separate.token_ids[0,:separate.lengths[0]].tolist())
            self.assertAlmostEqual(out.log_scores[i].item(),separate.log_scores[0].item(),places=6)
        self.assertTrue(any(len(x)>1 for x in m.reorders))
        self.assertTrue(all(len(set(sources.tolist())) == 1 for _,sources,_ in m.calls))
        self.assertFalse(any(x in (0,2) for i in range(2) for x in out.token_ids[i,:out.lengths[i]].tolist()))

    def test_completed_hypotheses_not_overwritten(self):
        out=beam_search(MockTransformer('eos_first'),self.src[:1],self.lengths[:1],self.mask[:1],beam_size=5,max_length=4,length_alpha=0)
        self.assertEqual(out.token_ids.tolist(),[[3]])
        self.assertTrue(out.finished[0])

    def test_length_normalization_and_raw_scores(self):
        for alpha in (0.0,0.6,1.0):
            out=beam_search(MockTransformer(),self.src,self.lengths,self.mask,beam_size=5,max_length=3,length_alpha=alpha)
            torch.testing.assert_close(out.scores,out.log_scores/out.lengths.to(torch.float64).pow(alpha))
            self.assertTrue(torch.isfinite(out.scores).all())

    def test_normalization_counts_eos_not_bos_or_padding(self):
        class FixedEOS(MockTransformer):
            def decode_step(self, token, state):
                _, new_state, _ = super().decode_step(token, state)
                logits = torch.full((len(token), 6), -2.0, device=token.device)
                for i, source in enumerate(state.source.tolist()):
                    next_id = 3 if source == 7 or new_state.prefix.shape[1] else 4
                    logits[i, next_id] = 2.0
                return logits, new_state, None

        for decoder in (greedy_decode, beam_search):
            for alpha in (0.0, 1.0):
                model = FixedEOS()
                result = decoder(model, self.src, self.lengths, self.mask,
                                 max_length=2, length_alpha=alpha)
                self.assertEqual(result.token_ids.tolist(), [[3, 0], [4, 3]])
                self.assertEqual(result.lengths.tolist(), [1, 2])
                self.assertEqual(result.finished.tolist(), [True, True])
                self.assertTrue(all(3 not in tokens.tolist() for tokens, _, _ in model.calls))
                logits = torch.full((6,), -2.0)
                logits[3] = 2.0
                first_score = float(logits.log_softmax(0)[3])
                logits[3], logits[4] = -2.0, 2.0
                content_score = float(logits.log_softmax(0)[4])
                eos_logits = torch.full((6,), -2.0)
                eos_logits[3] = 2.0
                eos_score = float(eos_logits.log_softmax(0)[3])
                self.assertAlmostEqual(result.log_scores[0].item(), first_score, places=7)
                self.assertAlmostEqual(result.log_scores[1].item(), content_score + eos_score, places=7)
                self.assertAlmostEqual(result.scores[0].item(), first_score, places=7)
                self.assertAlmostEqual(result.scores[1].item(), (content_score + eos_score) / (2 ** alpha), places=7)

    def test_normalization_can_change_winner(self):
        class LengthModel(MockTransformer):
            def decode_step(self, token, state):
                _, new_state, _ = super().decode_step(token, state)
                logits = torch.full((len(token), 6), -30.0)
                if new_state.prefix.shape[1] == 0:
                    logits[:, 3] = math.log(0.6)
                    logits[:, 4] = math.log(0.4)
                else:
                    logits[:, 4] = 10.0
                    logits[:, 3] = -10.0
                return logits, new_state, None
        raw = beam_search(LengthModel(), self.src[:1], self.lengths[:1], self.mask[:1],
                          beam_size=2, max_length=2, length_alpha=0)
        normalized = beam_search(LengthModel(), self.src[:1], self.lengths[:1], self.mask[:1],
                                 beam_size=2, max_length=2, length_alpha=1)
        self.assertEqual(raw.token_ids.tolist(), [[3]])
        self.assertEqual(normalized.token_ids.tolist(), [[4, 4]])
        self.assertGreater(raw.log_scores.item(), normalized.log_scores.item())
        self.assertGreater(normalized.scores.item(), raw.scores.item())

    def test_max_length_prefers_unfinished_when_better(self):
        out=beam_search(MockTransformer('never'),self.src[:1],self.lengths[:1],self.mask[:1],beam_size=5,max_length=2,length_alpha=0)
        self.assertEqual(out.token_ids.tolist(),[[4,4]])
        self.assertFalse(out.finished.item())

    def test_default_source_length_plus_50(self):
        src = torch.tensor([[7, 3, 0], [8, 9, 3]])
        out = greedy_decode(MockTransformer('never'), src, torch.tensor([2, 3]), src.eq(0))
        self.assertEqual(out.lengths.tolist(), [52, 53])
        self.assertEqual(out.finished.tolist(), [False, False])
        self.assertEqual(out.token_ids[0, 52].item(), 0)

    def test_beam_ties_are_deterministic(self):
        model = MockTransformer('ties')
        first = beam_search(model, self.src[:1], self.lengths[:1], self.mask[:1], beam_size=5, max_length=3)
        second = beam_search(MockTransformer('ties'), self.src[:1], self.lengths[:1], self.mask[:1], beam_size=5, max_length=3)
        self.assertTrue(torch.equal(first.token_ids, second.token_ids))
        self.assertTrue(torch.equal(first.scores, second.scores))
        self.assertEqual(first.token_ids.tolist(), [[4, 4, 4]])
        self.assertTrue(any(len(ids) > 1 and len(set(ids)) < len(ids) for ids in model.reorders))

    def test_invalid_parameters(self):
        model=MockTransformer()
        with self.assertRaises(ValueError): beam_search(model,self.src,self.lengths,self.mask,beam_size=0)
        with self.assertRaises(ValueError): greedy_decode(model,self.src,self.lengths,self.mask,max_length=0)
        with self.assertRaises(ValueError): greedy_decode(model,self.src,self.lengths,~self.mask,max_length=2)
        with self.assertRaises(ValueError): beam_search(model,self.src,self.lengths,self.mask,length_alpha=-1)
        model.training=True
        with self.assertRaisesRegex(ValueError,'eval'): greedy_decode(model,self.src,self.lengths,self.mask,max_length=2)

    def test_lstm_m2_integration(self):
        torch.manual_seed(13)
        model=Seq2SeqLSTM().eval()
        src=torch.tensor([[11,3,0],[7,9,3]])
        lengths=torch.tensor([2,3]); mask=src.eq(0)
        with torch.no_grad():
            state=model.init_decoder_state(model.encode(src,lengths,mask))
            _,new_state,_=model.decode_step(torch.tensor([2,2]),state)
            self.assertFalse(torch.equal(state.hidden,new_state.hidden))
            for decoder in (greedy_decode,beam_search):
                out=decoder(model,src,lengths,mask,max_length=3)
                self.assertEqual(out.token_ids.shape[0],2)
                self.assertTrue(bool(((out.token_ids>=0)&(out.token_ids<8000)).all()))
                self.assertTrue(bool((out.lengths<=3).all()))
                for row,n in zip(out.token_ids,out.lengths):
                    self.assertFalse(any(x in (0,2) for x in row[:n].tolist()))


if __name__ == '__main__': unittest.main(verbosity=2)
