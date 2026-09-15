from cs336_basics.train_bpe import train_bpe

if __name__ == "__main__":
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10_000, ["<|endoftext|>"])