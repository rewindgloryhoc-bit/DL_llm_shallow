import tempfile
from pathlib import Path
import multiprocessing
from tokenizer_parallel import get_workers
import cProfile
import pstats
import numpy as np
import os
from token_encoding import token_encode, token_decode


PROCESS_FULL = True

BASE_DIR = Path(__file__).resolve().parent
input_path2 = BASE_DIR / "owt_train.txt"
input_path1 = BASE_DIR / "TinyStoriesV2-GPT4-train.txt"

vocab_path1 = BASE_DIR / "token_data" / "vocabulary_TinyStories.pkl"
merges_path1 = BASE_DIR / "token_data" / "merges_TinyStories.pkl"
vocab_path2 = BASE_DIR / "token_data" / "vocabulary_owt_train.pkl"
merges_path2 = BASE_DIR / "token_data" / "merges_owt_train.pkl"

encoded_path1 = BASE_DIR / "token_data" / "encoded_TinyStories.pkl"
encoded_path2 = BASE_DIR / "token_data" / "encoded_owt_train.pkl"


def get_chunks_encode(workers, raw_text, file_size):
    chunk_len = file_size // workers
    child_seeker = list()
    start = 0
    with open(raw_text, "rb") as f:
        for i in range(workers):
            if i == workers - 1:
                child_seeker.append((start, file_size))
                break
            else:
                end = start + chunk_len
                f.seek(end)
                f.readline()
                end = min(f.tell(), file_size)
                child_seeker.append((start, end))
                start = end
    return child_seeker


#read offsets
def child_encode(args):
    chunk_idx, chunk_offsets, invert_vocab, merges_dict, temp_dir, special_token, custom_path = args
    with open(custom_path, "rb") as f:
        f.seek(chunk_offsets[0])
        b_chunk = f.read(chunk_offsets[1] - chunk_offsets[0])
        chunk_txt = b_chunk.decode("utf-8", errors="replace")

    coded_txt = np.array(token_encode(chunk_txt, invert_vocab, merges_dict), dtype=np.uint16)
    child_file_path = os.path.join(temp_dir, f'chunk{chunk_idx}.txt')
    coded_txt.tofile(child_file_path)
    return chunk_idx, len(coded_txt), child_file_path


def encode_parallel(custom_path, inverted_vocab, merges_dict, out_bin, special_token):
    workers = max(5, get_workers())
    f_size = os.path.getsize(custom_path)

    child_offsets = get_chunks_encode(workers, custom_path, f_size)

    Path(out_bin).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        worker_arg = [(i, chunk, inverted_vocab, merges_dict, temp_dir, special_token, custom_path)
                      for i, chunk in enumerate(child_offsets)]

        with multiprocessing.Pool(processes=workers) as pool:
            meta_data = list(pool.imap(child_encode, worker_arg))
        meta_data.sort(key=lambda x: x[0])
        shape = sum(length for _, length, _ in meta_data)
        final_memmap = np.memmap(
            out_bin,
            dtype=np.uint16,
            mode="w+",
            shape=shape,
        )

        offset = 0
        for idx, length, temp_path in meta_data:
            temp_data = np.fromfile(temp_path, dtype=np.uint16)
            final_memmap[offset:offset + length] = temp_data
            offset += length
            del temp_data
            os.remove(temp_path)
        final_memmap.flush()
    return out_bin, shape, f_size


def encode_user_prompt(prompt, inverted_vocab, merges_dict, special_token):
    encoded = token_encode(prompt, inverted_vocab, merges_dict, special_token)
    return encoded

