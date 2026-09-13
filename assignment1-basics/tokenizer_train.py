import tokenizer_parallel
import pickle


#a bpe object so save state for vocab dicionary and merges order.
class BPETokenizer:
    def __init__(self, input_path, vocab_size, vocab_path, merges_path, special_token = None):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.special_token = special_token
        self.vocab_dict = {}
        self.merges = []
        self.trained = False

    def bpe_train(self):
        if self.vocab_path.is_file() and self.merges_path.is_file():
            print("Vocabulary exists, no bpe training needed")
            print("do not call method 'save.bpe' since data are already saved")
            with open (self.vocab_path, 'rb') as vocab:
                self.vocab_dict = pickle.load(vocab)
            with open (self.merges_path, 'rb') as merges:
                self.merges = pickle.load(merges)
        #else if no data exist
        vocab_dict, merges = tokenizer_parallel.tokenization(self.vocab_size, self.input_path, self.special_token)
        self.trained = True
        self.vocab_dict = vocab_dict
        self.merges = merges


    def save_bpe(self):
        with open(self.vocab_path, 'wb') as v:
            pickle.dump(self.vocab_dict, v)
        with open(self.merges_path, 'wb') as m:
            pickle.dump(self.merges, m)

