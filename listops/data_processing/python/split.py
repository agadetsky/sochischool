# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Splits original data to create three files train.tsv, valid.tsv, test.tsv
import os
import numpy as np
from shutil import copyfile


rnd = np.random.RandomState(42)
with open("./listops/data_processing/raw/train_d20s.tsv") as f:
    lines = f.readlines()
    rnd.shuffle(lines)


if not os.path.exists("data_processing/listops/processed"):
    os.makedirs("data_processing/listops/processed")

with open("./listops/data_processing/processed/valid.tsv", 'w') as f:
    for line in lines[:1000]:
        f.write(line)
with open("./listops/data_processing/processed/train.tsv", 'w') as f:
    for line in lines[1000:]:
        f.write(line)

copyfile("./listops/data_processing/raw/test_d20s.tsv", "./listops/data_processing/processed/test.tsv")
