from cs336_basics.train_bpe import train_bpe
import json
import time

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    vocab, merges = train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    json_safe_vocab = {str(k): v.decode("latin-1") for k, v in vocab.items()}

    with open(file="owt-vocab.json", mode="w", encoding="utf-8") as f:
        json.dump(json_safe_vocab, f)

    with open(file="owt-merges.txt", mode="w", encoding="utf-8") as f:
        for pair in merges:
            p1 = pair[0].decode("latin-1")
            p2 = pair[1].decode("latin-1")
            f.write(f"{p1} {p2}\n")
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Script ran for {elapsed_time:.2f} seconds.")
