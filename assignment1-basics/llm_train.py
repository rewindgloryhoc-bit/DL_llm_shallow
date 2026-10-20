from pathlib import Path
from tokenizer_train import BPETokenizer
from tokenizer_encode import Encoder
from full_llm import Full_llm
from get_batch import run_get_batch
import torch
from cross_entropy import cross_entropy
from gradient_clipping import gradient_clipping
from check_point import save_checkpoint, load_checkpoint
import signal
import sys
from softmax import softmax
import tempfile
import time
from nucleus import nucleus_generation
from fused_optimizer import setup_optimizers, OPTIMIZER_CHOICES
import argparse
from experiment_report import write_report, default_experiment_id, sanitize_experiment_id
from mfu_benchmark import FullStepMFUBenchmarker


'''
this is a helper modeul which wraps the main llm archetiecture and optimizer,
and some other benchmark utilities.
'''



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

DATASET_CHOICES = ("tinystories", "opentext")

checkpoint_dir = BASE_DIR / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / "latest_model_checkpoint.pt"


def resolve_dataset(name):
    """Map --train name to input / vocab / merges / encoded paths."""
    if name == "tinystories":
        return {
            "name": name,
            "input_path": input_path1,
            "vocab_path": vocab_path1,
            "merges_path": merges_path1,
            "encoded_path": encoded_path1,
            "out_bin": encoded_path1,
            "vocab_size": 10000,
        }
    if name == "opentext":
        return {
            "name": name,
            "input_path": input_path2,
            "vocab_path": vocab_path2,
            "merges_path": merges_path2,
            "encoded_path": encoded_path2,
            "out_bin": encoded_path2,
            "vocab_size": 32000,
        }
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASET_CHOICES}")


step_size = 5000
TIME_BUDGET_SECONDS = 45 * 60
TRAIN_MODES = ("iter", "time")


def pick_device():
    """Baseline cloud/GPU training: prefer CUDA, otherwise fall back for local smoke tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device):
    # H100/A100: bf16 for most weights/activations. MPS/CPU stay fp32.
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


device = pick_device()
dtype = pick_dtype(device)


def d_ff_from_d_model(d_model):
    """SwiGLU width: ~8/3 d_model, floored to a multiple of 128."""
    return int((8 / 3 * d_model) // 128 * 128)


class LLM_train:
    def __init__(
            self, input_path, d_model, batch_size, num_heads, max_seq, theta, lr_min, lr_max, Tw, Tc, lamda, device, dtype, vocab_size, num_layer, dff=None, experiment_id=None, optimizer="muon", notes=None, decay_steps=1200, mode="iter", time_budget_seconds=TIME_BUDGET_SECONDS, Tw_sec=None, decay_sec=None, vocab_path=None, merges_path=None, encoded_path=None, dataset_name=None, grad_accum_steps=1):
        if mode not in TRAIN_MODES:
            raise ValueError(f"mode must be one of {TRAIN_MODES}, got {mode!r}")
        self.experiment_id = experiment_id or default_experiment_id()
        self.notes = notes
        self.optimizer_name = optimizer
        self.mode = mode
        self.time_budget_seconds = time_budget_seconds
        self.dataset_name = dataset_name
        self.input_path = input_path
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.encoded_path = encoded_path  # same role as out_bin
        self.d_model = d_model
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.num_heads = num_heads
        self.max_seq = max_seq
        self.theta = theta
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.Tw = Tw
        self.Tc = Tc
        self.decay_steps = decay_steps
        self.weight_decay = lamda
        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.current_step = 0
        self.layers = num_layer
        # Time mode: fixed-step warmup (Tw); decay is a fixed fraction of the budget.
        # Tw_sec is unused for scheduling (kept only if an explicit override is passed).
        self.Tw_sec = Tw_sec
        self.decay_sec = decay_sec if decay_sec is not None else time_budget_seconds / 9.0
        if dff is None:
            self.d_ff = d_ff_from_d_model(self.d_model)
        else:
            self.d_ff = dff
        self.transformer = Full_llm(
            self.d_model, self.num_heads, self.d_ff, self.max_seq, self.theta, self.layers, vocab_size, weights=None,
            device=device, dtype=dtype
        ).to(device)

        self.optimizer_muon, self.optimizer_adamw = setup_optimizers(
            self.transformer,
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            weight_decay=self.weight_decay,
            Tw=self.Tw,
            Tc=self.Tc,
            optimizer=self.optimizer_name,
            decay_steps=self.decay_steps,
            schedule_mode=self.mode,
            Tw_sec=self.Tw_sec,
            T_budget=self.time_budget_seconds,
            decay_sec=self.decay_sec,
        )
        self.optimizers = [opt for opt in (self.optimizer_muon, self.optimizer_adamw) if opt is not None]

        self.bpe_tokenizer = BPETokenizer(
            self.input_path, vocab_size, self.vocab_path, self.merges_path, special_token=["<|endoftext|>"]
        )
        self.bpe_tokenizer.bpe_train()
        if self.bpe_tokenizer.trained:
            self.bpe_tokenizer.save_bpe()
        self.encoder = Encoder.from_files(
            self.bpe_tokenizer.vocab_path, self.bpe_tokenizer.merges_path, special_tokens=["<|endoftext|>"]
        )


        self.save_requested = False
        self.train_start_time = None
        self.start_step = 0
        self.step_times = []
        self.loss_set = []
        self.lr_set = []
        self.mfu = None
        #directs to save checkpoint
        signal.signal(signal.SIGINT, self._handle_signal)


    def _handle_signal(self, sig, frame):
        print("training halted, prepare to save parameters and optimizer states")
        self.save_checkpoint()
        print("checkpoint saved")
        if self.train_start_time is not None:
            self.write_report(interrupted=True)
        sys.exit(0)

    def hyperparameters(self):
        if self.mode == "iter":
            schedule = "iteration-based (cos_annealing)"
        else:
            schedule = "hybrid (step warmup + time decay)"
        hp = {
            "schedule": schedule,
            "mode": self.mode,
            "optimizer": "AdamW only" if self.optimizer_name == "adamw" else "Muon (2D hidden) + AdamW (rest)",
            "lr_min": self.lr_min,
            "lr_max": self.lr_max,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_batch_size": self.batch_size * self.grad_accum_steps,
            "max_seq": self.max_seq,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.layers,
            "d_ff": self.d_ff,
            "vocab_size": self.vocab_size,
            "rope theta": self.theta,
            "device": self.device,
            "dtype": self.dtype,
            "dataset": self.dataset_name or Path(self.input_path).name,
            "vocab_path": Path(self.vocab_path).name,
            "encoded_path": Path(self.encoded_path).name,
        }
        if self.mode == "iter":
            hp["total steps"] = step_size
            hp["Tw (warmup steps)"] = self.Tw
            hp["Tc (cosine cycle steps)"] = self.Tc
            hp["decay_steps"] = self.decay_steps
            hp["Tw_sec / decay_sec / time_budget"] = "n/a (iter mode)"
        else:
            hp["time_budget_seconds"] = self.time_budget_seconds
            hp["time_budget_minutes"] = self.time_budget_seconds / 60
            hp["Tw (warmup steps)"] = self.Tw
            hp["decay_sec"] = self.decay_sec
            hp["decay_fraction_of_budget"] = self.decay_sec / self.time_budget_seconds
        return hp

    def write_report(self, interrupted=False):
        mfu_stats = self.mfu.report_stats() if self.mfu is not None else None
        report_path = write_report(
            self.experiment_id,
            self.train_start_time,
            time.time(),
            self.start_step,
            self.step_times,
            self.loss_set,
            self.hyperparameters(),
            interrupted=interrupted,
            notes=self.notes,
            mfu_stats=mfu_stats,
        )
        print(f"experiment report written to {report_path}")

    def save_checkpoint(self):
        save_checkpoint(
            self.transformer,
            self.optimizers,
            self.current_step,
            checkpoint_path
        )

    def load_checkpoint(self):
        start_step = load_checkpoint(checkpoint_path, self.transformer, self.optimizers)
        return start_step

    def forward(self):
        self.encoder.encode(self.input_path, self.encoded_path)
        print("encoding complete")

        if Path(checkpoint_path).is_file():
            self.current_step = self.load_checkpoint()

        # CUDA: Inductor. MPS/CPU: aot_eager. Outer-loop MFU timing stays outside compile.
        compile_backend = "inductor" if self.device.type == "cuda" else "aot_eager"
        self.transformer = torch.compile(self.transformer, backend=compile_backend)
        self.transformer.train()

        self.mfu = None
        if self.device.type == "cuda":
            self.mfu = FullStepMFUBenchmarker(
                self.d_model,
                self.d_ff,
                self.layers,
                self.dtype,
                activation_checkpointing=True,  # projections and SwiGLU recomputed; attention map kept
            )
            print(
                f"full-step MFU benchmark on (peak={self.mfu.peak_tflops:.0f} TFLOPS, "
                "warmup=15, window=20, credits recompute of projections and SwiGLU); "
                "compatible with torch.compile"
            )

        lr_set = self.lr_set
        loss_set = self.loss_set
        self.start_step = self.current_step
        self.train_start_time = time.time()
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["train_start_time"] = self.train_start_time
                group["schedule_mode"] = self.mode
        if self.device.type == "cuda":
            print(f"device={self.device} ({torch.cuda.get_device_name(self.device)}) compile={compile_backend}")
        else:
            print(f"device={self.device} compile={compile_backend}")
        if self.mode == "time":
            print(
                f"time mode (hybrid LR): budget={self.time_budget_seconds / 60:.0f} min, "
                f"Tw={self.Tw} steps (fixed), decay_sec={self.decay_sec:.1f} "
                f"({self.decay_sec / 60:.1f} min = 1/9 of budget)"
            )
        else:
            print(f"iter mode: training from step {self.current_step} to {step_size - 1}")

        step = self.current_step
        while True:
            if self.mode == "iter" and step >= step_size:
                break
            if self.mode == "time" and (time.time() - self.train_start_time) >= self.time_budget_seconds:
                elapsed = time.time() - self.train_start_time
                print(f"time budget reached ({elapsed / 60:.1f} min); stopping at step {step}")
                break

            if self.mode == "time" and step == self.Tw:
                elapsed = time.time() - self.train_start_time
                time_left = max(0.0, self.time_budget_seconds - elapsed)
                steady_sec = max(0.0, time_left - self.decay_sec)
                print(
                    f"warmup done at step {step} ({elapsed / 60:.2f} min elapsed); "
                    f"time left={time_left / 60:.2f} min → "
                    f"steady={steady_sec / 60:.2f} min @ lr_max, "
                    f"decay={self.decay_sec / 60:.2f} min"
                )

            step_start = time.perf_counter()
            self.current_step = step
            
            for opt in self.optimizers:
                opt.zero_grad()
                
            #the following implements gradient accumulation to save VRAM 
            accum_loss = 0.0
            for micro_step in range(self.grad_accum_steps):
                input_seq, target_label = run_get_batch(self.encoded_path, self.batch_size, self.max_seq, self.device)
                B, T = input_seq.shape[0], input_seq.shape[1]

                if self.mfu is not None:
                    self.mfu.record_start_fwd()
                    
                # Full_llm handles the chunked loss internally now to save memory
                loss_tensor = self.transformer(input_seq, targets=target_label, return_logits=False, grad_accum_steps=self.grad_accum_steps, mfu=self.mfu)
                loss_val = loss_tensor.item() / self.grad_accum_steps
                
                accum_loss += loss_val * self.grad_accum_steps
                
                if self.mfu is not None:
                    self.mfu.finish_step(B, T)

            accum_loss /= self.grad_accum_steps
            loss_set.append(accum_loss)
            gradient_clipping(self.transformer.parameters(), M=1)
            #actual update happens here.
            for opt in self.optimizers:
                opt.step()
            lr = self.optimizer_adamw.param_groups[0]["lr"]
            lr_set.append(lr)
            self.step_times.append(time.perf_counter() - step_start)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(
                f"[{stamp}] step={step} adamw_lr={lr:.6e} "
                f"loss={accum_loss:.4f}"
            )
            step += 1
        self.write_report()
        return self.transformer, lr_set, loss_set

    
    def text_generation(self, prompt=None, limit=256, p=0.91):
        self.load_checkpoint()
        nucleus_generation(self.encoder, self.transformer, self.device, self.max_seq, prompt, limit, p)






def dry_run(steps: int = 2, optimizer="muon"):
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
    d_ff = d_ff_from_d_model(d_model) or 128

    print(
        f"dry-run: d_model={d_model}, heads={num_heads}, seq={max_seq}, "
        f"batch={batch_size}, vocab={vocab_size}, layers={num_layers}, steps={steps}"
    )

    model = Full_llm(
        d_model, num_heads, d_ff, max_seq, theta, num_layers, vocab_size,
        weights=None, device=dry_device, dtype=dry_dtype,
    ).to(dry_device)
    optimizer_muon, optimizer_adamw = setup_optimizers(
        model,
        lr_min=1e-3,
        lr_max=3e-3,
        weight_decay=0.01,
        Tw=1,
        Tc=10,
        optimizer=optimizer,
    )
    optimizers = [opt for opt in (optimizer_muon, optimizer_adamw) if opt is not None]
    model.train()

    for step in range(steps):
        input_seq = torch.randint(0, vocab_size, (batch_size, max_seq), device=dry_device)
        target_label = torch.randint(0, vocab_size, (batch_size, max_seq), device=dry_device)
        loss = model(input_seq, targets=target_label, return_logits=False)
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss}"
        for opt in optimizers:
            opt.zero_grad()
        # loss.backward() is called internally now
        gradient_clipping(model.parameters(), M=1)
        for opt in optimizers:
            opt.step()
        muon_lr = f"{optimizer_muon.param_groups[0]['lr']:.6e}" if optimizer_muon is not None else "n/a"
        print(
            f"dry-run step {step}: loss={loss.item():.4f} "
            f"adamw_lr={optimizer_adamw.param_groups[0]['lr']:.6e} "
            f"muon_lr={muon_lr}"
        )

    with tempfile.TemporaryDirectory(dir=checkpoint_dir) as tmp:
        ckpt = Path(tmp) / "dry_run_checkpoint.pt"
        save_checkpoint(model, optimizers, iteration=steps, out=ckpt)
        load_checkpoint(ckpt, model, optimizers)
        print(f"dry-run checkpoint ok: {ckpt.name}")

    print("dry-run passed")


def main(experiment_id=None, optimizer="muon", notes=None, mode="iter", dataset="opentext"):
    paths = resolve_dataset(dataset)
    print(f"dataset={paths['name']} input={paths['input_path'].name} vocab_size={paths['vocab_size']}")

    # --- schedule knobs ---
    # Warmup is fixed in steps (independent of wall clock).
    # In --mode time: decay uses 1/9 of the 45 min budget; the rest after warmup is lr_max hold.
    time_budget_seconds = TIME_BUDGET_SECONDS  # 45 min
    Tw = 300
    decay_sec = time_budget_seconds / 10.0  # 5 min final cosine decay
    grad_accum_steps = 1  # micro-batches per optimizer step (effective batch = batch_size * this)

    training = LLM_train(
        input_path=paths["input_path"],
        d_model=768,
        batch_size=256,
        num_heads=12,
        max_seq=512,
        theta=10000,
        lr_min=1.5e-4,
        lr_max=1.5e-3,
        Tw=Tw,
        Tc=5000,
        decay_steps=1000,
        lamda=0.1,
        device=device,
        dtype=dtype,
        vocab_size=paths["vocab_size"],
        num_layer=8,
        experiment_id=experiment_id,
        optimizer=optimizer,
        notes=notes,
        mode=mode,
        time_budget_seconds=time_budget_seconds,
        decay_sec=decay_sec,
        vocab_path=paths["vocab_path"],
        merges_path=paths["merges_path"],
        encoded_path=paths["out_bin"],
        dataset_name=paths["name"],
        grad_accum_steps=grad_accum_steps,
    )
    transformer, lr_set, loss_set = training.forward()
    # Loss curve is already written by experiment_report; skip interactive
    # plt.show() so headless cloud GPUs do not hang waiting for a display.

def text_generation(prompt=None, limit=256, p=0.91, dataset="opentext"):
    paths = resolve_dataset(dataset)
    training = LLM_train(
        input_path=paths["input_path"],
        d_model=512,
        batch_size=32,
        num_heads=16,
        max_seq=256,
        theta=10000,
        lr_min=1.5e-4,
        lr_max=1.5e-3,
        Tw=500,
        Tc=5000,
        lamda=0.1,
        device=device,
        dtype=dtype,
        vocab_size=paths["vocab_size"],
        num_layer=8,
        vocab_path=paths["vocab_path"],
        merges_path=paths["merges_path"],
        encoded_path=paths["out_bin"],
        dataset_name=paths["name"],
    )

    training.text_generation(prompt, limit, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--text-generation", action="store_true")
    parser.add_argument("--experiment", "-e", default=None,
                        help="experiment ID used as the report title and folder name under experiments/")
    parser.add_argument("--optimizer", "-o", choices=OPTIMIZER_CHOICES, default="muon",
                        help="muon: Muon for 2D hidden matrices + AdamW for the rest; adamw: AdamW for everything")
    parser.add_argument("--notes", "-n", nargs="+", default=None,
                        help="free-text notes recorded under 'Brief summary' in report.md")
    parser.add_argument("--mode", choices=TRAIN_MODES, default="iter",
                        help="iter: stop after a fixed step count; time: stop after a wall-clock budget")
    parser.add_argument("--train", choices=DATASET_CHOICES, default="opentext",
                        help="tinystories or opentext: selects input, vocab, merges, and encoded paths")
    args = parser.parse_args()
    notes = " ".join(args.notes) if args.notes else None

    if args.dry_run:
        dry_run(optimizer=args.optimizer)
    elif args.text_generation:
        text_generation(dataset=args.train)
    else:
        experiment_id = sanitize_experiment_id(args.experiment) if args.experiment else None
        main(experiment_id, optimizer=args.optimizer, notes=notes, mode=args.mode, dataset=args.train)
