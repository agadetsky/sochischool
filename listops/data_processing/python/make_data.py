# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this is a script for generating data for extrapolation task


import argparse
import random
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--min_depth', type=int, default=1)
parser.add_argument('--max_depth', type=int, default=20)
parser.add_argument('--train_data_points', type=int, default=2000)
parser.add_argument('--valid_data_points', type=int, default=2000)
parser.add_argument('--test_data_points', type=int, default=2000)
parser.add_argument('--min_length', type=int, default=2)
parser.add_argument('--train_max_length', type=int, default=100)
parser.add_argument('--valid_max_length', type=int, default=150)
parser.add_argument('--test_max_length', type=int, default=np.inf)
parser.add_argument('--folder_name', type=str)
parser.add_argument('--no_sm', default=False, action='store_true') # No SUM_MOD operator

args = parser.parse_args()

BASE_PATH = "./listops/data/"

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
END = "]"

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25
MAX_ARGS = 5


def generate_tree(depth, max_depth):
    if depth < max_depth:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1, max_depth))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'
        else:
            return to_string(t[0], parens) + ' ' + to_string(t[1], parens)


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])

def save_dataset(fname, data_points, min_length, max_length, min_depth, max_depth):
    with open(os.path.join(BASE_PATH, args.folder_name, fname + ".tsv"), "w") as f:
        data = set()
        while len(data) < data_points:
            example = generate_tree(1, max_depth)
            tokens = to_string(example, parens=False).split()
            example_length = len(tokens)
            depth = get_depth_from_tokens(tokens)
            if ((min_length <= example_length < max_length)
                    and (min_depth <= depth < max_depth)):
                print('Added.', example_length, len(data))
                data.add(example)
                example_str = (str(to_value(example)) +
                               '\t' + to_string(example, parens=True) + '\n')
                f.write(example_str)

def get_depth_from_tokens(tokens):
    max_d = 0
    current_d = 0
    for t in tokens:
        if '[' in t:
            current_d +=1
            max_d = max(current_d, max_d)
        elif ']' == t:
            current_d -=1
        else:
            continue
    return max_d

def main():
    np.random.seed(args.seed)
    if args.no_sm:
        OPERATORS.remove(SUM_MOD)
    print('Operators', OPERATORS)
    # Make dataset folder if it doesn't exist.
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, args.folder_name), exist_ok=True)

    # We want to ensure that there is no overlap between train, valid and test.
    total_data = set()
    for fname in ["train", "valid", "test"]:
        if fname == "train":
            data_points = args.train_data_points
            max_length = args.train_max_length
        elif fname == "valid":
            data_points = args.valid_data_points
            max_length = args.valid_max_length
        else:
            data_points = args.test_data_points
            max_length = args.test_max_length
        min_length = args.min_length
        min_depth = args.min_depth
        # +1 because of how they defined their generation + how we measure depth
        max_depth = args.max_depth + 1
        print(f"Generating {fname} data with {data_points} samples.")
        with open(os.path.join(BASE_PATH, args.folder_name, fname + ".tsv"), "w") as f:
            data = set()
            while len(data) < data_points:
                example = generate_tree(1, max_depth)
                tokens = to_string(example, parens=False).split()
                example_length = len(tokens)
                depth = get_depth_from_tokens(tokens)
                if ((min_length <= example_length < max_length)
                        and (min_depth <= depth < max_depth)
                        and (example not in total_data)):
                    print('Added.', example_length, len(data))
                    data.add(example)
                    total_data.add(example)
                    example_str = (str(to_value(example)) +
                                '\t' + to_string(example, parens=True) + '\n')
                    f.write(example_str)


if __name__ == '__main__':
    main()
