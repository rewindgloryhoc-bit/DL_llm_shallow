from pathlib import Path
from softmax import softmax
import torch
import os
import random



BASE_DIR = Path(__file__).resolve().parent
sample_path = BASE_DIR / "TinyStoriesV2-GPT4-valid.txt"




def get_sample_text(sample_path):
    file_size = os.path.getsize(sample_path)
    random_start = random.randint(0, file_size - 1)
    chunk_size = 128
    with open(sample_path, 'rb') as f:
        f.seek(random_start)
        chunk = f.read(chunk_size)
    return chunk.decode('utf-8', errors='replace')



def nucleus_generation(encoder, transformer, device, max_seq, prompt=None, limit=256, p=0.91):
    if prompt is None:
        prompt = get_sample_text(sample_path)   
    encoded_prompt = encoder.encode_prompt(prompt)
    encoded_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    transformer.eval()
    with torch.no_grad():
        i = 0
        while True:
            x = transformer(encoded_tensor)
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
            new_word = encoder.decode([token_int])
            if new_word == '<|endoftext|>':
                break
            print(new_word, end=' ', flush=True)
            i += 1
            if i >= limit:
                print("generation limit reached, end of text generation")
                break
            #stack the encoded tensor on seq dimension
            encoded_tensor = torch.cat([encoded_tensor, token], dim=-1).to(device)
            if encoded_tensor.size(-1) > max_seq:
                print("max sequence length reached, end of text generation")
                break
        return f"generated text of length{i}"