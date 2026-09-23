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
import tempfile
import time
import matplotlib.pyplot as plt
from nucleus import nucleus_generation





BASE_DIR = Path(__file__).resolve().parent
input_path2 = BASE_DIR / "owt_train.txt"
input_path1 = BASE_DIR / "TinyStoriesV2-GPT4-train.txt"

vocab_path1 = BASE_DIR / "token_data" / "vocabulary_TinyStories.pkl"
merges_path1 = BASE_DIR / "token_data" / "merges_TinyStories.pkl"
vocab_path2 = BASE_DIR / "token_data" / "vocabulary_owt_train.pkl"
merges_path2 = BASE_DIR / "token_data" / "merges_owt_train.pkl"

#change this path later, should not be in pkl. earlier legacy to be corrected
encoded_path1 = BASE_DIR / "token_data" / "encoded_TinyStories.pkl"
encoded_path2 = BASE_DIR / "token_data" / "encoded_owt_train.pkl"

out_bin = encoded_path1

checkpoint_dir = BASE_DIR / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / "latest_model_checkpoint.pt"


step_size = 5000
device = torch.device('mps')
dtype = torch.float32


class LLM_train:
    def __init__(
            self, input_path, d_model, batch_size, num_heads, max_seq, theta, lr_min, lr_max, Tw, Tc, lamda, device, dtype, vocab_size, num_layer, dff=None):
        self.input_path = input_path
        self.d_model = d_model
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq = max_seq
        self.theta = theta
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.Tw = Tw
        self.Tc = Tc
        self.weight_decay = lamda
        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.current_step = 0
        self.layers = num_layer
        if dff is None:
            self.d_ff = int((8 / 3 * self.d_model) // 64 * 64)
        else:
            self.d_ff = dff
        self.transformer = Full_llm(
            self.d_model, self.num_heads, self.d_ff, self.max_seq, self.theta, self.layers, vocab_size, weights=None,
            device=device, dtype=dtype
        ).to(device)

        self.optimizer = Adam_W(
            self.transformer.parameters(),
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            weight_decay=self.weight_decay,
            Tw=self.Tw,
            Tc=self.Tc,
        )

        self.bpe_tokenizer = BPETokenizer(
            self.input_path, vocab_size, vocab_path1, merges_path1, special_token=["<|endoftext|>"]
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
            self.current_step,
            checkpoint_path
        )

    def load_checkpoint(self):
        start_step = load_checkpoint(checkpoint_path, self.transformer, self.optimizer)
        return start_step

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
            self.current_step = self.load_checkpoint()
        self.transformer = torch.compile(self.transformer, backend="aot_eager")
        self.transformer.train()

        lr_set = []
        loss_set = []
        for step in range(self.current_step, step_size):
            self.current_step = step
            input_seq, target_label = run_get_batch(out_bin, self.batch_size, self.max_seq, self.device)
            logits = self.transformer(input_seq)
            loss = cross_entropy(logits, target_label)
            loss_set.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            gradient_clipping(self.transformer.parameters(), M=1)
            self.optimizer.step()
            lr = self.optimizer.param_groups[0]["lr"]
            lr_set.append(lr)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{stamp}] step={step} lr={lr:.6e} loss={loss.item():.4f}")
        return self.transformer, lr_set, loss_set

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

    def text_generation(self, prompt=None, limit=256, p=0.91):
        self.load_checkpoint()
        nucleus_generation(self.encoder, self.transformer, self.device, self.max_seq, prompt, limit, p)






def dry_run(steps: int = 2):
    """Synthetic forward/backward smoke test — no tokenizer or real dataset."""
    dry_device = torch.device("cpu")
    dry_dtype = torch.float32
    d_model = 64
    num_heads = 4
    max_seq = 32
    batch_size = 2
    vocab_size = 128
    num_layers = 2
    theta = 10000.0
    d_ff = int((8 / 3 * d_model) // 64 * 64) or 128

    print(
        f"dry-run: d_model={d_model}, heads={num_heads}, seq={max_seq}, "
        f"batch={batch_size}, vocab={vocab_size}, layers={num_layers}, steps={steps}"
    )

    model = Full_llm(
        d_model, num_heads, d_ff, max_seq, theta, num_layers, vocab_size,
        weights=None, device=dry_device, dtype=dry_dtype,
    ).to(dry_device)
    optimizer = Adam_W(model.parameters(), lr_min=1e-3, Tw=1, Tc=10, lr_max=3e-3)
    model.train()

    for step in range(steps):
        input_seq = torch.randint(0, vocab_size, (batch_size, max_seq), device=dry_device)
        target_label = torch.randint(0, vocab_size, (batch_size, max_seq), device=dry_device)
        logits = model(input_seq)
        assert logits.shape == (batch_size, max_seq, vocab_size), logits.shape
        loss = cross_entropy(logits, target_label)
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss}"
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), M=1)
        optimizer.step()
        print(f"dry-run step {step}: loss={loss.item():.4f}")

    with tempfile.TemporaryDirectory(dir=checkpoint_dir) as tmp:
        ckpt = Path(tmp) / "dry_run_checkpoint.pt"
        save_checkpoint(model, optimizer, iteration=steps, out=ckpt)
        load_checkpoint(ckpt, model, optimizer)
        print(f"dry-run checkpoint ok: {ckpt.name}")

    print("dry-run passed")


def main():
    training = LLM_train(
        input_path=input_path1,
        d_model=512,
        batch_size=64,
        num_heads=16,
        max_seq=256,
        theta=10000,
        lr_min=8e-4,
        lr_max=8e-3,
        Tw=500,
        Tc=5000,
        lamda=0.1,
        device=device,
        dtype=dtype,
        vocab_size=10000,
        num_layer=4,
        dff=1344,
    )
    transformer, lr_set, loss_set = training.forward()
    plt.plot(lr_set, loss_set)
    plt.xlabel('lr')
    plt.ylabel('loss')
    plt.title('lr vs loss')
    plt.show()

def text_generation(prompt=None, limit=256, p=0.91):
    training = LLM_train(
    input_path=input_path1,
    d_model=512,
    batch_size=64,
    num_heads=16,
    max_seq=256,
    theta=10000,
    lr_min=8e-4,
    lr_max=8e-3,
    Tw=500,
    Tc=5000,
    lamda=0.1,
    device=device,
    dtype=dtype,
    vocab_size=10000,
    num_layer=4,
    dff=1344,
    )

    training.text_generation(prompt, limit, p)


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run()
    elif "--text-generation" in sys.argv:
        text_generation()
    else:
        main()
