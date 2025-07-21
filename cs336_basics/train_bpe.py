import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
from multiprocessing import Pool, cpu_count


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

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


def pre_tokenize(chunk, escapedSpecialTokens):
    delimitedChunks = re.split("|".join(escapedSpecialTokens), chunk)
    text_matched_list = []
    for delimitedChunk in delimitedChunks:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, delimitedChunk)
        for match in matches:
            text_matched = tuple(match.group())
            text_matched_list.append(text_matched)
    return text_matched_list


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    freqTable = defaultdict(int)  # int: bytes
    merges = set()  # (token1, token2)[]
    tokenCounter = 256
    vocab = {i: chr(i).encode("utf8") for i in range(256)}  # int: bytes
    currentVocab = set()
    escapedSpecialTokens = [re.escape(token) for token in special_tokens]

    chunksToTokenize = []

    with open(input_path, "rb") as f:
        num_processes = cpu_count()
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8")
        )

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunksToTokenize.append(chunk)

    with Pool(processes=num_processes) as pool:
        text_matched_list_pool = pool.starmap(
            pre_tokenize, [(chunk, escapedSpecialTokens) for chunk in chunksToTokenize]
        )

    for text_matched_list in text_matched_list_pool:
        for text_matched in text_matched_list:
            freqTable[text_matched] += 1

    while len(vocab) < vocab_size:
        subFreqTable = defaultdict(int)
        for key, value in freqTable.items():
            for i in range(len(key)):
                if i != 0:
                    pair = key[i - 1] + key[i]
                    subFreqTable[pair] += value

        subFreqTableList = list(subFreqTable.items())
        subFreqTableList.sort(key=lambda x: x[1], reverse=True)
        keyToMerge = subFreqTableList[0][0]
        newKeys = []
        keysToDelete = set()
        print(subFreqTableList[:5])
        for key, value in freqTable.items():
            for i in range(len(key)):
                if i != 0 and key[i - 1] == keyToMerge[0] and key[i] == keyToMerge[1]:
                    newKey = key[: i - 1] + (keyToMerge,) + key[i + 1 :]
                    if keyToMerge not in currentVocab:
                        vocab[tokenCounter] = keyToMerge.encode("utf8")
                        currentVocab.add(keyToMerge)
                        tokenCounter += 1
                    merges.add(
                        (keyToMerge[0].encode("utf8"), keyToMerge[1].encode("utf8"))
                    )
                    newKeys.append((newKey, value, keyToMerge))
                    keysToDelete.add(key)
                    if tokenCounter == vocab_size:
                        return (vocab, list(merges))

        for keyToDelete in keysToDelete:
            del freqTable[keyToDelete]
        for newKey in newKeys:
            freqTable[newKey[0]] = value


if __name__ == "__main__":
    train_bpe(
        "/Users/yifanlim/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
        500,
        ["<|endoftext|>"],
    )
