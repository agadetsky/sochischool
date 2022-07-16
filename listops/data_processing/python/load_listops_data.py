import os
# Loads raw tsv files and creates dictionaries

T_SHIFT = 0
T_REDUCE = 1

def load_listops_data(data_path, maxlen=float("inf")):
    # Check for vocabulary (if it exists)
    vocab_path = os.path.join(os.path.dirname(data_path), 'vocab.txt')
    vocab_exists = os.path.exists(vocab_path)
    if vocab_exists:
        idx_to_word, word_to_idx = load_vocab_dicts(vocab_path)

    data = []
    with open(data_path) as f:
        too_long = 0
        too_short = 0
        for e_id, line in enumerate(f):
            label, seq = line.strip().split('\t')
            e = dict()
            e["label"] = int(label)
            e["sentence"] = seq
            e["tokens"], e["transitions"] = convert_bracketed_sequence(seq.split(' '))
            if len(e["tokens"]) > maxlen:
                too_long += 1
                continue
            if len(e["tokens"]) == 1:
                too_short += 1
                continue
            if vocab_exists:
                e["num_tokens"] = [word_to_idx[e] for e in e["tokens"]]
            e["id"] = str(e_id)
            data.append(e)
    print(f"file path: {data_path}")
    print(f"number of skipped sentences due to length > {maxlen}: {too_long}")
    print(f"number of skipped sentences due to length < 2: {too_short}")
    return data

def convert_bracketed_sequence(seq):
    tokens, transitions = [], []
    if len(seq) == 1:
        return seq, []
    for item in seq:
        if item == "(":
            continue
        if item == ")":
            transitions.append(T_REDUCE)
        else:
            tokens.append(item)
            transitions.append(T_SHIFT)
    return tokens, transitions

def load_vocab_dicts(vocab_path):
    with open(vocab_path) as f:
        idx_to_word = [word.strip() for word in f.readlines()]
        word_to_idx = dict()
        for idx, word in enumerate(idx_to_word):
            word_to_idx[word] = idx
    return idx_to_word, word_to_idx
