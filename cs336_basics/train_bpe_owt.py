import pickle
from cs336_basics.train_bpe import train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe("data/owt_train.txt", 32_000, ["<|endoftext|>"])
    with open("owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
