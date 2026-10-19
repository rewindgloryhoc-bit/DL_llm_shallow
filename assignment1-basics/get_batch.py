import torch
import numpy as np
import time
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

#prompt format
current_time = time.localtime()
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)

tokenized_prompt_DIR = BASE_DIR / "token_data"
tokenized_prompt_DIR.mkdir(parents=True, exist_ok=True)
tokenized_prompt = BASE_DIR / "token_data" / f'prompt_at_{formatted_time}'

# Reuse memmaps across microbatches; opening / remapping every call is wasteful.
_MEMMAP_CACHE: dict[str, np.memmap] = {}


#check if the file is a pickle file, for future use.
#if it is a pickle file, convert it to a raw uint16 file.
def _is_pickle_file(path: Path) -> bool:
    """Peek only the first byte — never read the whole corpus into RAM."""
    with open(path, "rb") as f:
        return f.read(1) == b"\x80"


#convertion func to load the file in memmap format.
def _ensure_raw_uint16(path: Path) -> None:
    """One-time convert of legacy pickled ndarray dumps to raw uint16."""
    #if file does not exist or is not a pickle file(already converted), return.
    if not path.is_file() or not _is_pickle_file(path):
        return
    with open(path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=np.uint16)
    #avoids leaving a half-written corpus if the process crashes mid-tofile.
    tmp = path.with_name(path.name + ".rawtmp")
    arr.tofile(tmp)
    tmp.replace(path)


#populates the cache if not already present.
def _get_memmap(encoded_path) -> np.memmap:
    path = Path(encoded_path)
    key = str(path.resolve())
    #returns none if not cached.
    cached = _MEMMAP_CACHE.get(key)
    if cached is not None:
        return cached
    #convert the file to raw uint16 format.
    _ensure_raw_uint16(path)
    #create a memmap object for the file.
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    _MEMMAP_CACHE[key] = mm
    return mm


def run_get_batch(
    encoded_path, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_data = _get_memmap(encoded_path)
    seq_len = len(raw_data)
    max_idx = seq_len - context_length - 1
    starts = np.random.randint(0, max_idx + 1, size=batch_size)

    # Contiguous slices per row (memmap-friendly); avoid Python list + np.array copy chain.
    x_np = np.empty((batch_size, context_length), dtype=np.int64)
    y_np = np.empty((batch_size, context_length), dtype=np.int64)
    for i, k in enumerate(starts):
        x_np[i] = raw_data[k : k + context_length]
        y_np[i] = raw_data[k + 1 : k + 1 + context_length]

    # Embedding / gather need integer index dtypes; uint16 from memmap is not allowed.
    torch_device = torch.device(device) if not isinstance(device, torch.device) else device
    if torch_device.type == "cuda":
        x = torch.from_numpy(x_np).pin_memory().to(torch_device, non_blocking=True)
        y = torch.from_numpy(y_np).pin_memory().to(torch_device, non_blocking=True)
    else:
        x = torch.from_numpy(x_np).to(torch_device)
        y = torch.from_numpy(y_np).to(torch_device)
    return x, y
