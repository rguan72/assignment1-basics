from collections.abc import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.vocab_idx = dict((bytes, id) for id, bytes in vocab.items())
        self.merges = merges
        self.merges_idx = dict((byte_pair, idx) for idx, byte_pair in enumerate(merges))
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        pass

    def encode(self, text: str) -> list[int]:
        # TODO: special tokens & chunk boundaries
        # if self.special_tokens:
        #     sections =  re.split(pattern="|".join([re.escape(special_token) for special_token in self.special_tokens]), string=text)
        # else:
        #     sections = [text]
        tokens = []
        for match in re.finditer(PAT, text):
            match_encoded = match.group().encode("utf-8")
            pretoken = [match_encoded[i:i+1] for i in range(len(match_encoded))]
            merges_remain = True
            while merges_remain:
                merges_remain = False
                earliest_merge_idx: int | float = float('inf')
                for i in range(len(pretoken)-1):
                    byte_pair = (pretoken[i],pretoken[i+1])
                    if (byte_pair in self.merges_idx) and (self.merges_idx[byte_pair] < earliest_merge_idx):
                        earliest_merge_idx = self.merges_idx[byte_pair]
                if earliest_merge_idx != float('inf'):
                    pair = self.merges[earliest_merge_idx]
                    merges_remain = True
                    pretoken_out = []
                    i = 0
                    while i < len(pretoken):
                        if i+1 < len(pretoken) and (pretoken[i], pretoken[i+1]) == pair:
                            pretoken_out.append(pretoken[i]+pretoken[i+1])
                            i += 2
                        else:
                            pretoken_out.append(pretoken[i])
                            i += 1
                    pretoken = pretoken_out
            for merged_bytes in pretoken:
                tokens.append(self.vocab_idx[merged_bytes])
        return tokens
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[id] for id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")