from collections.abc import Iterable, Iterator
import regex as re
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.vocab_idx = dict((bytes, id) for id, bytes in vocab.items())
        self.merges = merges
        self.merges_idx = dict((byte_pair, idx) for idx, byte_pair in enumerate(merges))
        self.special_tokens = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for section in self._split_on_special_tokens(text):
            if section in self.special_tokens:
                tokens.append(self.vocab_idx[section.encode("utf-8")])
                continue
            for match in re.finditer(PAT, section):
                pretoken = [bytes([b]) for b in match.group().encode("utf-8")]
                tokens.extend(self.vocab_idx[piece] for piece in self._apply_merges(pretoken))
        return tokens
    
    def _split_on_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(map(re.escape, ordered)) + ")"
        return [s for s in re.split(pattern, text) if s]
    
    def _apply_merges(self, pretoken: list[bytes]) -> list[bytes]:
        while True:
            candidates = (p for p in zip(pretoken, pretoken[1:]) if p in self.merges_idx)
            pair = min(candidates, key=self.merges_idx.__getitem__, default=None)
            if pair is None:
                return pretoken
    
            merged: list[bytes] = []
            i = 0
            while i < len(pretoken):
                if i + 1 < len(pretoken) and (pretoken[i], pretoken[i + 1]) == pair:
                    merged.append(pretoken[i] + pretoken[i + 1])
                    i += 2
                else:
                    merged.append(pretoken[i])
                    i += 1
            pretoken = merged
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[id] for id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")