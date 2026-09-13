import torch
import numpy as np
import time
from llm_train import BASE_DIR


#prompt format
current_time = time.localtime()
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)

tokenized_prompt_DIR = BASE_DIR / "token_data"
tokenized_prompt_DIR.mkdir(parents=True, exist_ok=True)
tokenized_prompt = BASE_DIR / "token_data" / f'prompt_at_{formatted_time}'



def run_get_batch(
    encoded_path, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_data = np.memmap(encoded_path, dtype=np.uint16, mode="r")
    seq_len = len(raw_data)
    max_idx = seq_len - context_length - 1
    start_indices = np.random.randint(0, max_idx + 1, size=batch_size)
    input_seq = [raw_data[k:k+context_length] for k in start_indices]
    label = [raw_data[k+1:k+context_length+1] for k in start_indices]
    inputseq = torch.as_tensor(input_seq, device=device)
    label = torch.as_tensor(label, device=device)
    return inputseq, label




