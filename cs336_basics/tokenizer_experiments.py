from cs336_basics.tokenizer import Tokenizer
import random
import numpy as np
import pickle
import time

endoftext = "<|endoftext|>"


def iter_documents(f, chunk_size: int = 1 << 20):
    """Yield one document at a time (each ending in <|endoftext|>) from a text file,
    reading in fixed-size chunks so the whole file is never held in memory.
    Splitting on the special token (rather than on newlines) gives the same tokens
    as encode() on the whole file, since encode() splits on special tokens anyway."""
    buffer = ""
    while chunk := f.read(chunk_size):
        buffer += chunk
        *docs, buffer = buffer.split(endoftext)
        for doc in docs:
            yield doc + endoftext
    if buffer:
        yield buffer

tinystories_tokenizer: Tokenizer = Tokenizer.from_files("tinystories_vocab.pkl", "tinystories_merges.pkl", [endoftext])
owt_tokenizer: Tokenizer = Tokenizer.from_files("owt_vocab.pkl", "owt_merges.pkl", [endoftext])

random.seed(42)

with open("data/TinyStoriesV2-GPT4-valid.txt") as of:
    tinystories_validation_docs = of.read().split(endoftext)
    tinystories_sampled = random.sample(tinystories_validation_docs, 10)

with open("data/owt_valid.txt") as of:
    owt_validation_docs = of.read().split(endoftext)
    owt_sampled = random.sample(owt_validation_docs, 10)

tinystories_bytes = 0
tinystories_tokens = 0
for sample in tinystories_sampled:
    sample_bytes = len(sample.encode("utf-8"))
    tinystories_bytes += sample_bytes
    sample_tokens = len(tinystories_tokenizer.encode(sample))
    tinystories_tokens += sample_tokens
tinystories_compression_ratio = tinystories_bytes / tinystories_tokens
print(f"tinystories compression ratio: {tinystories_compression_ratio}")

owt_bytes = 0
owt_tokens = 0
for sample in owt_sampled:
    sample_bytes = len(sample.encode("utf-8"))
    owt_bytes += sample_bytes
    sample_tokens = len(owt_tokenizer.encode(sample))
    owt_tokens += sample_tokens
owt_compression_ratio = owt_bytes / owt_tokens
print(f"owt compression ratio: {owt_compression_ratio}")

owt_cross_bytes = 0
owt_cross_tokens = 0
for sample in owt_sampled:
    sample_bytes = len(sample.encode("utf-8"))
    owt_cross_bytes += sample_bytes
    sample_tokens = len(tinystories_tokenizer.encode(sample))
    owt_cross_tokens += sample_tokens
owt_cross_compression_ratio = owt_cross_bytes / owt_cross_tokens
print(f"owt_cross compression ratio: {owt_cross_compression_ratio}")

print("tokenizing tinystories validation")
start_time = time.perf_counter()
with open("data/TinyStoriesV2-GPT4-valid.txt") as tf:
    tinystories_valid_tokenized = np.fromiter(tinystories_tokenizer.encode_iterable(iter_documents(tf)), dtype="uint16")
    with open("tinystories_valid_tokenized.pkl", "wb") as f:
        pickle.dump(tinystories_valid_tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
end_time = time.perf_counter()
print(f"tinystories validation time: {end_time-start_time}")

print("tokenizing tinystories train")
start_time = time.perf_counter()
with open("data/TinyStoriesV2-GPT4-train.txt") as tf:
    tinystories_train_tokenized = np.fromiter(tinystories_tokenizer.encode_iterable(iter_documents(tf)), dtype="uint16")
    with open("tinystories_train_tokenized.pkl", "wb") as f:
        pickle.dump(tinystories_train_tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
end_time = time.perf_counter()
print(f"tinystories train time: {end_time-start_time}")

print("tokenizing owt validation")
start_time = time.perf_counter()
with open("data/owt_valid.txt") as tf:
    owt_valid_tokenized = np.fromiter(owt_tokenizer.encode_iterable(iter_documents(tf)), dtype="uint16")
    with open("owt_valid_tokenized.pkl", "wb") as f:
        pickle.dump(owt_valid_tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
end_time = time.perf_counter()
print(f"owt validation time: {end_time-start_time}")

print("tokenizing owt train")
start_time = time.perf_counter()
with open("data/owt_train.txt") as tf:
    owt_train_tokenized = np.fromiter(owt_tokenizer.encode_iterable(iter_documents(tf)), dtype="uint16")
    with open("owt_train_tokenized.pkl", "wb") as f:
        pickle.dump(owt_train_tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
end_time = time.perf_counter()
print(f"owt train time: {end_time-start_time}")