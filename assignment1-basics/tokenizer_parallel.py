from functools import partial
from collections import Counter, defaultdict
import regex as re
import heapq
import multiprocessing
import cProfile
import pstats
import gc
import os


'''
    Frequency dictionary is {tupe of ints: counts} to save memory space
    thus, pair_wise_map is {(pair of ints): counts},
    heap is in {(pair of ints): counts}
    inverted dict is in {(pairs of ints): list of ints}
'''

def get_workers():
    return min(6, multiprocessing.cpu_count())


def get_chunks_frequency(input_path, num_workers):
    file_size = os.path.getsize(input_path)
    chunk_size = file_size // num_workers
    child_offsets = []
    start = 0
    with open(input_path, "rb") as f:
        for i in range(num_workers):
            if i == num_workers - 1:
                child_offsets.append((start, file_size))
            else:
                end = start + chunk_size
                f.seek(end)
                f.readline()
                end = min(f.tell(), file_size)
                child_offsets.append((start, end))
                start = end
        return child_offsets


def child_frequency(input_path, offsets:tuple, special_token):
    with open(input_path, "rb") as f:
        f.seek(offsets[0])
        chunk = f.read(offsets[1] - offsets[0])
        chunk = chunk.decode("utf-8", errors="replace")
    frequency_dict = {}
    if special_token == []:
        preprocessed = [chunk]
    else:
        #add escape character, note that sp
        special_es =  [re.escape(j) for j in special_token]
        special_or = r"(" + r"|".join(special_es) + r")"
        preprocessed = re.split(special_or, chunk)
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for piece in preprocessed:
        if piece in special_token:
            continue
        for x in re.findall(PAT, piece):
            #this converts bytes object into tuple of ints
            processed = tuple(x.encode("utf-8"))
            if processed in frequency_dict:
                frequency_dict[processed] += 1
            else:
                frequency_dict[processed] = 1
    return frequency_dict


def parallel_frequency(input_path: str, special_token):
    frequency_map = Counter({})
    num_workers = get_workers()
    chunks_offsets = get_chunks_frequency(input_path, num_workers)
    worker_func = partial(child_frequency, input_path, special_token=special_token)
    with multiprocessing.Pool(num_workers) as pool:
        for result in pool.imap(worker_func, chunks_offsets):
            frequency_map.update(result)
    return frequency_map


def pair_heap_init(pair_wise_map: dict):
    pair_wise_heap = []
    for pair, count in pair_wise_map.items():
        pair_wise_heap.append((-count, pair))
    heapq.heapify(pair_wise_heap)
    return pair_wise_heap


#note that all keys in pair_wise_map are tuple of integer to save memory space
def inverted_pairs_init(frequency_dict):
    inverted_pairs = defaultdict(set)
    pair_wise_map = Counter()
    for word_tuple, count in frequency_dict.items():
        for m in range(len(word_tuple) - 1):
            pair = (word_tuple[m], word_tuple[m+1])
            inverted_pairs[pair].add(word_tuple)
            pair_wise_map[pair] += count
    pair_wise_heap = pair_heap_init(pair_wise_map)
    return inverted_pairs, pair_wise_map, pair_wise_heap


def vocab_matrix_init(special_token):
    vocabulary_matrix = {}
    for i in range(0,256):
        vocab = bytes([i])
        vocabulary_matrix[i] = vocab
    for token in special_token:
        token_bytes = token.encode('utf-8')
        vocabulary_matrix[len(vocabulary_matrix)] = token_bytes
    return vocabulary_matrix


#small helper function to merge a token.
#evaluates to None when heap is empty
#note pre-merge is in (int, int)
def merge_token(pair_wise_heap, pair_wise_map):
    while pair_wise_heap:
        neg_count, pre_merge = heapq.heappop(pair_wise_heap)
        current_count = pair_wise_map.get(pre_merge, 0)

        if current_count == -neg_count and current_count > 0:
            return pre_merge, current_count
    return None, None


#quick look up
def merge_new_map(pre_merge, frequency_map, pair_wise_map, pair_wise_heap, vocabulary_matrix, inverted_pairs):
    new_token = len(vocabulary_matrix)
    vocabulary_matrix[new_token] = vocabulary_matrix[pre_merge[0]] + vocabulary_matrix[pre_merge[1]]
    reduced_target = list(inverted_pairs[pre_merge])
    temp_heap = set()
    for word_tuple in reduced_target:
        count = frequency_map.pop(word_tuple)
        word_len = len(word_tuple)
        #decrement/delete old pairs
        for i in range(word_len - 1):
            old_pair = (word_tuple[i], word_tuple[i + 1])
            pair_wise_map[old_pair] -= count
            temp_heap.add(old_pair)
            if old_pair in inverted_pairs:
                inverted_pairs[old_pair].discard(word_tuple)
                if not inverted_pairs[old_pair]:
                    del inverted_pairs[old_pair]
        b = 0
        temp_key = []
        while b < word_len:
            if b < word_len - 1 and word_tuple[b] == pre_merge[0] and word_tuple[b + 1] == pre_merge[1]:
                temp_key.append(new_token)
                b += 2
            else:
                temp_key.append(word_tuple[b])
                b += 1
        new_freq = tuple(temp_key)
        #assumes a previous merge yielded same result
        if new_freq in frequency_map:
            frequency_map[new_freq] += count
        else:
            frequency_map[new_freq] = count

        for i in range(len(new_freq) - 1):
            new_pair = (new_freq[i], new_freq[i+1])
            temp_heap.add(new_pair)
            if new_pair in pair_wise_map:
                pair_wise_map[new_pair] += count
            else:
                pair_wise_map[new_pair] = count
            inverted_pairs[new_pair].add(new_freq)
    #optimizing heap push
    for pairs in temp_heap:
        h_count = pair_wise_map[pairs]
        if h_count <= 0:
            del pair_wise_map[pairs]
            continue
        heapq.heappush(pair_wise_heap, (-h_count, pairs))
    return vocabulary_matrix, frequency_map


def tokenization(vocab_size, input_path, special_token: list):
    vocab_dict = vocab_matrix_init(special_token)
    merges = []
    frequency_map = parallel_frequency(input_path, special_token)
    inverted_pairs, pair_wise_map, pair_wise_heap = inverted_pairs_init(frequency_map)
    gc.collect()
    heap_iter = 0
    while len(vocab_dict) < vocab_size:
        if heap_iter > 2000:
            pair_wise_heap = None
            pair_wise_heap = pair_heap_init(pair_wise_map)
            heap_iter = 0
        if not pair_wise_heap:
            print("no more merge to complete, end of tokenization")
            return vocab_dict, merges
        pre_merge, count = merge_token(pair_wise_heap, pair_wise_map)
        if pre_merge is None:
            print("no more heap entry, end of tokenization")
            return vocab_dict, merges
        merges.append(pre_merge)
        vocab_dict, frequency_map = merge_new_map(pre_merge, frequency_map, pair_wise_map, pair_wise_heap, vocab_dict, inverted_pairs)
        heap_iter += 1
    return vocab_dict, merges

'''
if __name__ == '__main__':
    input_path = "./owt_train.txt"
    profiler = cProfile.Profile()
    profiler.enable()

    vocab_dict, merges = tokenization(32000, input_path, special_token=["<|endoftext|>"])
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile.prof")
    print("Profiling successful! Saved to profile.prof")
    print(vocab_dict)
'''