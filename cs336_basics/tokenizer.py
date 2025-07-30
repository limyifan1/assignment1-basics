from collections.abc import Iterable, Iterator
from collections import defaultdict
from cs336_basics.pre_tokenize import pre_tokenize
from multiprocessing import Pool, cpu_count
import regex as re

class Tokenizer:
    # int_to_byte_vocab: dict[int, bytes], e.g. { 5: b't'}
    # byte_to_int_vocab: dict[bytes, int], e.g. { b't':5}
    # merges: set[tuple[bytes, bytes]], {(b't', b'h')}
    # special_tokens: list[str] | None = None,
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.int_to_byte_vocab = vocab
        self.byte_to_int_vocab = {value: key for key, value in vocab.items()}
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return cls()
    
    def generate_pairs_set(self, key_bytes: bytes) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for i in range(len(key_bytes)):
            if i == 0:
                continue
            pairs.add((key_bytes[i-1], key_bytes[i]))
        return pairs
        
    def encode(self, text: str) -> list[int]:
        pre_tokens = self.pre_tokenize(text)
        ans: list[int] = []
        for pre_token in pre_tokens:
            pairs_set = self.generate_pairs_set(pre_token)
            pre_token_stack: list[bytes] = [bytes([pre_token_byte]) for pre_token_byte in pre_token]
            for pair_to_merge in self.merges:
                if pair_to_merge in pairs_set:
                    new_pre_token_stack: list[bytes]  = []
                    for i in range(len(pre_token)):
                        if i != 0 and new_pre_token_stack[-1] == pair_to_merge[0] and pre_token[i] == pair_to_merge[1]:
                            new_pre_token_stack[-1] = new_pre_token_stack[-1] + bytes([pre_token[i]])
                        else:
                            new_pre_token_stack.append(bytes([pre_token[i]]))
                    pre_token_stack = new_pre_token_stack
                pairs_set = self.generate_pairs_set(b''.join(pre_token_stack))
            for byte in pre_token_stack:
                ans.append(self.byte_to_int_vocab[byte])
        return ans

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield self.encode(text)
    
    def pre_tokenize(self, text: str) -> list[bytes]:
        escaped_special_tokens = [re.escape(token) for token in self.special_tokens] 
        delimitedChunks = re.split("|".join(escaped_special_tokens), text)
        pre_tokens = []
        for delimitedChunk in delimitedChunks:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            matches = re.finditer(PAT, delimitedChunk)
            for match in matches:
                text_utf = match.group().encode("utf-8")
                pre_tokens.append(text_utf)
        return pre_tokens

    def decode(self, ids: list[int]) -> str:
        arr = bytearray()
        for id in ids:
            if isinstance(id, int):
                arr.extend(self.int_to_byte_vocab[id])
        return arr.decode('utf-8', errors='replace')