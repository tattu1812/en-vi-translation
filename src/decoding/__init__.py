"""Architecture-neutral inference for models implementing the decoding protocol."""
from .greedy import greedy_decode
from .beam_search import beam_search
from .interface import DecodeResult

__all__ = ["DecodeResult", "greedy_decode", "beam_search"]
