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
_PINNED_BATCH_CACHE = {}


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

    #allocate memory for the batch on cpu directly instead of using numpy empty arrays.
    cache_key = (batch_size, context_length)
    if cache_key not in _PINNED_BATCH_CACHE:
        batch_x = torch.empty((batch_size, context_length), dtype=torch.long)
        batch_y = torch.empty((batch_size, context_length), dtype=torch.long)
        torch_device = torch.device(device) if not isinstance(device, torch.device) else device
        if torch_device.type == "cuda":
            batch_x = batch_x.pin_memory()
            batch_y = batch_y.pin_memory()
        _PINNED_BATCH_CACHE[cache_key] = (batch_x, batch_y)

    batch_x, batch_y = _PINNED_BATCH_CACHE[cache_key]

    #in place copy of the data to the batch.(memmap -> torch tensor)
    for i, k in enumerate(starts):
        batch_x[i].copy_(torch.from_numpy(raw_data[k : k + context_length].astype(np.int64)))
        batch_y[i].copy_(torch.from_numpy(raw_data[k + 1 : k + 1 + context_length].astype(np.int64)))

    torch_device = torch.device(device) if not isinstance(device, torch.device) else device
    if torch_device.type == "cuda":
        x = batch_x.to(torch_device, non_blocking=True)
        y = batch_y.to(torch_device, non_blocking=True)
    else:
        x = batch_x.to(torch_device)
        y = batch_y.to(torch_device)
        
    return x, y
