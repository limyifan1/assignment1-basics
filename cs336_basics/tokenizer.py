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
    
    def encode(self, text: str) -> list[int]:
        pre_tokens = self.pre_tokenize(text)
        encoded = []
        for pre_token_byte in pre_tokens:
            byte_stack: list[bytes] = []
            for next_int in pre_token_byte:
                next_byte = bytes([next_int])
                if byte_stack:
                    top_byte = byte_stack[-1]
                    byte_pair = (top_byte, next_byte)
                    if byte_pair in self.merges:
                        byte_stack[-1] = top_byte + next_byte
                        continue
                byte_stack.append(next_byte)
            for byte_item in byte_stack:
                encoded.append(self.byte_to_int_vocab[byte_item])
        return encoded
    
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
            arr.extend(self.int_to_byte_vocab[id])
        return arr.decode('utf-8', errors='replace')