
import regex as re
import multiprocessing
import heapq
import multiprocessing

def get_workers():
    return multiprocessing.cpu_count()

def get_chunks(input: str, num_workers: int):
    text_length = len(input) // num_workers
    for i in range(num_workers):
        if i == num_workers - 1:
            yield text[i * text_length:]
        else:
            yield text[i * text_length:(i + 1) * text_length]


def child_incre_vocab(chunk_piece: str):
    sub_dict = {}
    encoded_piece = chunk_piece.encode("utf-8")
    for byte in encoded_piece:
        if byte in sub_dict:
            sub_dict[byte] += 1
        else:
            sub_dict[byte] = 1
    return sub_dict


def parallel_pair_wise(input: str):
    num_workers = get_workers()
    chunks_generator = get_chunks(input, num_workers)
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.imap(child_incre_vocab, chunks_generator)

        master_dict = {}
        for result in results:
            master_dict.update(result)
    return master_dict



def vocab_matrix_init(special_token):
    vocabulary_matrix = {}
    for i in range(0,256):
        vocab = bytes([i])
        vocabulary_matrix[vocab] = i
    for token in special_token:
        token_bytes = token.encode('utf-8')
        vocabulary_matrix[token_bytes] = len(vocabulary_matrix)
    return vocabulary_matrix


def build_frequency_map(input_str: str, special_token:list):
    special_token_p = r"(" + "|".join(re.escape(t) for t in special_token) + r")"
    preprocessed = re.split(special_token_p, input_str)
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #GPT regex
    frequency_map = {}
    #for loop to build frequency map
    for text in preprocessed:
        if not text:
            continue
        if text in special_token:
            continue
        for x in re.findall(PAT, text):
            processed = tuple(bytes([k]) for k in x.encode("utf-8"))
            if processed in frequency_map:
                frequency_map[processed] += 1
            else:
                frequency_map[processed] = 1
    return frequency_map


def build_pair_wise(frequency_map: dict):
    #implement merge pre-step
    pair_wise_map = {}
    for word_tuple, count in frequency_map.items():
        if len(word_tuple) == 1:
            continue
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            if pair in pair_wise_map:
                pair_wise_map[pair] += count
            else:
                pair_wise_map[pair] = count
    pair_wise_heap = [(-count, pair) for pair, count in pair_wise_map.items()]
    heapq.heapify(pair_wise_heap)
    return pair_wise_map, pair_wise_heap


def merge_token(pair_wise_heap):
    counts, pre_merge = heapq.heappop(pair_wise_heap)
    new_token = b"".join(pre_merge)
    return pre_merge, new_token, counts


def merge_new_map(frequency_map: dict, pre_merge:bytes, new_token: bytes, vocab_dict, pair_wise_heap, pair_wise_map):
    new_map = {}
    for word_tuple, count in frequency_map.items():
        temp_key = []
        i = 0
        has_pre_merge = False
        for x in range(len(word_tuple) - 1):
            if word_tuple[x] == pre_merge[0] and word_tuple[x + 1] == pre_merge[1]:
                has_pre_merge = True
        if not has_pre_merge:
            new_map[word_tuple] = count
            continue
        if len(word_tuple) > 2:
            for j in range(len(word_tuple) - 1):
                old_pair = (word_tuple[j], word_tuple[j + 1])
                pair_wise_map[old_pair] = max(0, pair_wise_map[old_pair] - count)
                heapq.heappush(pair_wise_heap, (-pair_wise_map[old_pair], old_pair))
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == pre_merge[0] and word_tuple[i + 1] == pre_merge[1]:
                temp_key.append(new_token)
                i += 2
            else:
                temp_key.append(word_tuple[i])
                i += 1
        new_key = tuple(temp_key) #ensure new key is wrapped in tuple, avoid edge case
        new_map[new_key] = count
        if len(new_key) >= 2:
            for k in range(len(new_key) - 1):
                new_pair = (new_key[k], new_key[k + 1])
                if new_pair not in pair_wise_map:
                    pair_wise_map[new_pair] = count
                else:
                    pair_wise_map[new_pair] += count
                # never push a single byte into heap
                heapq.heappush(pair_wise_heap, (-pair_wise_map[new_pair], new_pair))
    vocab_dict[new_token] = len(vocab_dict)
    frequency_map = new_map
    return frequency_map, vocab_dict


def tokenization(vocab_size: int, input_path: str = None, special_token = None):
    special_token = ["<|endoftext|>"]
    vocabulary_dict = vocab_matrix_init(special_token)
    input_str = ("This code demonstrates that you are thinking about how to merge the winning pair together, but it will break because of two main structural issues: modifying a dictionary while looping over it, and treating a tuple like a string.")
    frequency_map = build_frequency_map(input_str, special_token)
    pair_wise_map, pair_wise_heap = build_pair_wise(frequency_map)
    print(len(pair_wise_map))
    while len(vocabulary_dict) <= vocab_size:
        if pair_wise_map == {}:
            print("no more pairs to iterate, end merge")
            break
        pre_merge, new_token, counts = merge_token(pair_wise_heap)
        frequency_map, vocabulary_dict = merge_new_map(frequency_map, pre_merge, new_token, vocabulary_dict, pair_wise_heap, pair_wise_map)
    return vocabulary_dict, pair_wise_map, frequency_map

vocab, merges, frequency_map = tokenization(500)
print(vocab)
print(frequency_map)
