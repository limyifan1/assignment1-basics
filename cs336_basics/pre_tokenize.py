from collections import defaultdict
import regex as re


def pre_tokenize(chunk, escapedSpecialTokens):
    delimitedChunks = re.split("|".join(escapedSpecialTokens), chunk)
    text_matched_dict = defaultdict(int)
    for delimitedChunk in delimitedChunks:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = re.finditer(PAT, delimitedChunk)
        for match in matches:
            text_utf = match.group().encode("utf-8")
            text_matched = tuple(bytes([b]) for b in text_utf)
            text_matched_dict[text_matched] += 1
    return text_matched_dict
