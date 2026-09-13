from pathlib import Path
from tokenizer_train import BPETokenizer
from tokenizer_encode import Encoder
import cProfile
import pstats
from full_llm import Full_llm
from get_batch import run_get_batch
import torch
from cross_entropy import cross_entropy
from Adam_W import Adam_W
from gradient_clipping import gradient_clipping
from check_point import save_checkpoint, load_checkpoint
import signal
import sys
from softmax import softmax


BASE_DIR = Path(__file__).resolve().parent
input_path2 = BASE_DIR / "owt_train.txt"
input_path1 = BASE_DIR / "TinyStoriesV2-GPT4-train.txt"

vocab_path1 = BASE_DIR / "token_data" / "vocabulary_TinyStories.pkl"
merges_path1 = BASE_DIR / "token_data" / "merges_TinyStories.pkl"
vocab_path2 = BASE_DIR / "token_data" / "vocabulary_owt_train.pkl"
merges_path2 = BASE_DIR / "token_data" / "merges_owt_train.pkl"

encoded_path1 = BASE_DIR / "token_data" / "encoded_TinyStories.pkl"
encoded_path2 = BASE_DIR / "token_data" / "encoded_owt_train.pkl"

out_bin = encoded_path1

checkpoint_dir = BASE_DIR / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / "latest_model_checkpoint.pt"


num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
lr = 3e-5
Tw = num_epochs // 10
Tc = num_epochs  - Tw


class LLM_train:
    def __init__(
            self, input_path, d_model, batch_size, num_heads, max_seq, theta, lr, beta, lamda, device, dtype, vocab_size, num_layer):
        self.input_path = input_path
        self.d_model = d_model
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq = max_seq
        self.theta = theta
        self.lr = lr
        self.beta = beta
        self.lamda = lamda
        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.current_epoch = 0
        self.layers=num_layer

        d_ff = (8 / 3 * self.d_model) // 64 * 64
        self.transformer = Full_llm(
            self.d_model, self.num_heads, d_ff, self.max_seq, self.theta, self.layers, vocab_size, weights=None,
            device=device, dtype=dtype
        ).to(device)

        self.optimizer = Adam_W(self.transformer.parameters(), lr=self.lr, weight_decay=self.lamda)

        self.bpe_tokenizer = BPETokenizer(
            self.input_path, vocab_size, vocab_path2, merges_path2, special_token=["<|endoftext|>"]
        )
        self.bpe_tokenizer.bpe_train()
        if self.bpe_tokenizer.trained:
            self.bpe_tokenizer.save_bpe()
        vocab_path = self.bpe_tokenizer.vocab_path
        merges_path = self.bpe_tokenizer.merges_path

        self.encoder = Encoder.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])


        self.save_requested = False
        #directs to save checkpoint
        signal.signal(signal.SIGINT, self._handle_signal)


    def _handle_signal(self, sig, frame):
        print("training halted, prepare to save parameters and optimizer states")
        self.save_checkpoint()
        print("checkpoint saved")
        sys.exit(0)

    def save_checkpoint(self):
        save_checkpoint(
            self.transformer,
            self.optimizer,
            self.current_epoch,
            checkpoint_path
        )

    def load_checkpoint(self):
        start_epoch = load_checkpoint(checkpoint_path, self.transformer, self.optimizer)
        return start_epoch

    def forward(self):
        #profiler block to check encoding
        profiler = cProfile.Profile()
        profiler.enable()
        self.encoder.encode(self.input_path, out_bin)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats("profile.prof")
        print("encoding complete")

        if Path(checkpoint_path).is_file():
            self.current_epoch = self.load_checkpoint()
        self.transformer.train()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            input_seq, target_label = run_get_batch(out_bin, self.batch_size, self.max_seq, self.device)
            logits = self.transformer(input_seq)
            loss = cross_entropy(logits, target_label)
            self.optimizer.zero_grad()
            loss.backward()
            gradient_clipping(self.transformer.parameters(), M=1)
            self.optimizer.step()
            print(f"Loss: {loss.item():.4f}")
        return self.transformer

    '''
    #naive text generation
    def text_generation(self, limit=None):
        prompt = input('enter prompt for text generation:')
        encoded_prompt = self.encoder.encode_prompt(prompt)
        encoded_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=self.device)
        self.transformer.eval()
        with torch.no_grad():

            if limit is not None:
                i = 0
                while i < limit:
                    x = self.transformer(encoded_tensor)
                    # last row should be in shape [vocab_size,]
                    last_row = x[0, -1, :]
                    pi = softmax(last_row, i=-1)
                    new_token = torch.argmax(pi, dim=-1).item()
                    new_word = self.encoder.decode(new_token)
                    if new_word == '<|endoftext|>':
                        break
                    print(new_word, end=' ', flush=True)
                    encoded_prompt.append(new_token)
                    encoded_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=self.device)

    '''

    def nucleus_generation(self, limit=None, p=0.91):
        prompt = input('enter prompt for text generation:')
        encoded_prompt = self.encoder.encode_prompt(prompt)
        encoded_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=self.device)
        self.transformer.eval()
        with torch.no_grad():
            i = 0
            while True:
                x = self.transformer(encoded_tensor)
                #last row is [,vocab_size]
                last_row = x[0, -1, :]
                sorted_row, indices = torch.sort(last_row, descending=True)
                pi = softmax(sorted_row, i=-1)
                sum_ps = 0
                for idx, val in enumerate(pi):
                    sum_ps += val.item()
                    if sum_ps > p:
                        break
                last_idx = idx + 1
                sorted_row[last_idx:] = float('-inf')
                pi_normalized = softmax(sorted_row, i=-1)
                #sample index should be in [,1]
                # add batch dimension
                sampled_index = torch.multinomial(pi_normalized, 1)
                token_int = indices[sampled_index.item()].item()
                #avoid scalar indexing here
                token = indices[sampled_index].unsqueeze(0)
                new_word = self.encoder.decode(token_int)
                if new_word == '<|endoftext|>':
                    break
                print(new_word, end=' ', flush=True)
                i += 1
                if limit is not None and i >= limit:
                    break
                #stack the encoded tensor on seq dimension
                encoded_tensor = torch.cat([encoded_tensor, token], dim=-1).to(self.device)
            return f"generated text of length{i}"


def main():
    training = LLM_train(
        input_path=input_path1,
        d_model=512,
        batch_size=32,
        num_heads=16,
        max_seq=256,
        theta=10000,
        lr=lr,
        beta=None,
        lamda=None,
        device=device,
        dtype=dtype,
        vocab_size=10000,
        num_layer=4
    )

    training.forward()

