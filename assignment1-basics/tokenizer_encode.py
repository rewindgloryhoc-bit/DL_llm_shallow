
import pickle
from token_encoding import token_encode, token_decode
import tokenization_script


'''
so merges is in list of tuple of paired ints.
vocab is in {token_id: bytes}
bytes merge translate merges into (byte, byte)
finally we need the order of merges to process encoding
'''

class Encoder:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverted = {entry: rank for rank, entry in vocab.items()}
        self.bytes_merge = [(vocab[x[0]], vocab[x[1]]) for x in merges]
        self.merge_dict = {pair: rank for rank, pair in enumerate(self.bytes_merge)}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def encode(self, input_path, out_bin):
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            out_bin, len_encoded, len_bytes = tokenization_script.encode_parallel(
                input_path, self.inverted, self.merge_dict, out_bin, sorted_specials
            )
        else:
            out_bin, len_encoded, len_bytes = tokenization_script.encode_parallel(
                input_path, self.inverted, self.merge_dict, out_bin, self.special_tokens
            )
        compression_ratio = len_bytes / len_encoded
        print(f"compression ratio is {compression_ratio} for {input_path}")


    def encode_prompt(self, prompt):
        return tokenization_script.encode_user_prompt(prompt, self.inverted, self.merges, self.special_tokens)


    def decode(self, ids):
        return token_decode(ids, self.vocab)