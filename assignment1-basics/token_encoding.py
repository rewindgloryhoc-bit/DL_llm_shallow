import regex as re
import heapq


def find_merge(word_tuple, merges, inverted_dict):
    pairs_counter = set()
    m_heap = []
    encoded = []
    for i in range(len(word_tuple) - 1):
        pairs = (word_tuple[i], word_tuple[i+1])
        if pairs in merges:
            pairs_counter.add(pairs)
            m_heap.append((merges[pairs], pairs))
    heapq.heapify(m_heap)

    while m_heap and len(word_tuple) > 1:
        val, pair = heapq.heappop(m_heap)
        new_word = []
        x = 0
        while x < len(word_tuple):
            if x < len(word_tuple) - 1 and word_tuple[x] == pair[0] and word_tuple[x+1] == pair[1]:
                merged = b''.join(pair)
                new_word.append(merged)
                x += 2
            else:
                new_word.append(word_tuple[x])
                x += 1
        #to guard stale pairs, where popped heap pair is no longer valid
        if new_word == word_tuple:
            continue

        for k in range(len(new_word) - 1):
            new_pair = (new_word[k], new_word[k+1])
            if new_pair in pairs_counter:
                continue
            if new_pair in merges:
                pairs_counter.add(new_pair)
                heapq.heappush(m_heap, (merges[new_pair], new_pair))
        word_tuple = new_word
    for i in word_tuple:
        encoded.append(inverted_dict[i])
    return encoded

#bottonm encoding function used in child_encode serialization task.
def token_encode(inputs:str, inverted_dict, merges, special_token = None):
    t_encode = []
    if not special_token:
        pre_processed = [inputs]
        special_token = []
    else:
        special_token_p = r"(" + "|".join(re.escape(t) for t in special_token) + r")"
        pre_processed = re.split(special_token_p, inputs)
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for piece in pre_processed:
        if not piece:
            continue
        if piece in special_token:
            t_encode.append(inverted_dict[piece.encode('utf-8')])
            continue
        #avoid single character piece use generic list comprehension
        for x in re.findall(PAT, piece):
            s = [bytes([k]) for k in x.encode('utf-8')]
            t_encode += find_merge(s, merges, inverted_dict)
    return t_encode


def token_decode(sequence, vocab_dict):
    t_decode = []
    for x in sequence:
        decoded = vocab_dict.get(x, None)
        if decoded:
            t_decode.append(decoded)
        else:
            print(f"{x} is not valid token in vocabulary")
    c_decode = b"".join(t_decode)
    f_decoded = c_decode.decode('utf-8', errors='replace')
    return f_decoded
