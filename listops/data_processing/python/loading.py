import numpy as np
import os

T_SHIFT = 0
T_REDUCE = 1

word_to_ix = {
'0': 0,
'1': 1,
'2': 2,
'3': 3,
'4': 4,
'5': 5,
'6': 6,
'7': 7,
'8': 8,
'9': 9,
'[MAX': 10,
'[MED': 11,
'[MIN': 12,
'[SM': 13,
']': 14,
'<PAD>': 15 # void as never appears in raw_data
}

ix_to_word = {v: k for k, v in word_to_ix.items()}

def load_raw_data(data_path, maxlen):
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
            e["num_tokens"] = [word_to_ix[e] for e in e["tokens"]]
            e['hp_ix'] = hpix_from_tokens(e["tokens"])
            e['hp_arcmat'] = hparcmat_from_hpix(e['hp_ix'])
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

def hpix_from_tokens(tokens):
    # for the hp-tree I treat the first symbol as the root
    pointers = []
    stack = []
    current_head = -1 # -1 indicates symbol points to root
    for i, token in enumerate(tokens):
        pointers.append(current_head)
        if token == ']' and stack:
            current_head = stack.pop()
        elif not token.isnumeric(): # is operator
            stack.append(current_head)
            current_head = i
    return pointers

def hparcmat_from_hpix(hpix):
    assert hpix[0] == -1 # is root symbol
    n = len(hpix)
    eye = np.eye(n)
    arcmat = np.zeros((n, n))
    cols = eye[:, hpix[1:]]
    arcmat[:, 1:] = cols
    assert arcmat[:, 0].sum() == 0
    assert np.all(arcmat[:, 1:].sum(0) == 1)
    return arcmat
