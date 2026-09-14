from git import PathLike
import os
from typing import BinaryIO
import collections
import regex as re
from multiprocessing import Pool
import copy

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(input_path: str | PathLike, start: int, end: int, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        pretokens: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        sections =  re.split(pattern="|".join([re.escape(special_token) for special_token in special_tokens]), string=chunk)
        for section in sections:
            for match in re.finditer(PAT, section):
                match_encoded = match.group().encode("utf-8")
                match_bytes = tuple(match_encoded[i:i+1] for i in range(len(match_encoded)))
                pretokens[match_bytes] += 1
        return pretokens

def train_bpe(input_path: str | PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # TODOS
    # pretokenization efficiency: handle parallelization
    # merge efficiency: don't have to merge over the entire vocab? potentially?
    num_processes = 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    args_list = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args_list.append((input_path, start, end, special_tokens))
    with Pool(processes=num_processes) as pool:
        pretoken_chunks = pool.starmap(pretokenize, args_list)

    pretokens = copy.deepcopy(pretoken_chunks[0])
    for pretoken_chunk in pretoken_chunks[1:]:
        for pretoken, count in pretoken_chunk.items():
            pretokens[pretoken] += count

    merges: list[tuple[bytes, bytes]] = []
    vocab = create_initial_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        pairs = get_stats(pretokens)
        max_pair = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(max_pair)
        pretokens = merge_vocab(max_pair, pretokens)
        new_idx = len(vocab)
        vocab[new_idx] = max_pair[0] + max_pair[1]

    return vocab, merges

def create_initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = int.to_bytes(i)
    for j, special_token in enumerate(special_tokens):
        vocab[256 + j] = special_token.encode("utf-8")
    return vocab

def get_stats(vocab: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pairs = collections.defaultdict(int)
    for pretoken, freq in vocab.items():
        for i in range(len(pretoken) - 1):
            pairs[pretoken[i],pretoken[i+1]] += freq
    return pairs

def merge_vocab(pair: tuple[bytes, bytes], v_in: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
    v_out: dict[tuple[bytes, ...], int] = {}
    for pretoken in v_in:
        pretoken_out = ()
        i = 0
        while i < len(pretoken):
            if i+1 < len(pretoken) and (pretoken[i], pretoken[i+1]) == pair:
                pretoken_out += (pretoken[i]+pretoken[i+1],)
                i += 2
            else:
                pretoken_out += (pretoken[i],)
                i += 1
        v_out[pretoken_out] = v_in[pretoken]
    return v_out

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))