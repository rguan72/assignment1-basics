import pickle
from cs336_basics.train_bpe import train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10_000, ["<|endoftext|>"])
    with open("tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
