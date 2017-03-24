from argparse import ArgumentParser
from os import listdir
from os.path import join
from warnings import warn
from collections import defaultdict

import numpy as np
from scipy import stats

parser = ArgumentParser("Compare two simulation directories for discrepancies.")
parser.add_argument("dir1")
parser.add_argument("dir2")
parser.add_argument("--subset", action = "store_true", default = False)
args = parser.parse_args()

def file_to_lengths(full_path):
    lengths = defaultdict(list)
    with open(full_path, "r") as labeled_file:
        for line in labeled_file:
            # If the program crashed, the output can be left in an
            # inconsistent state.
            try:
                unlabeled_id, shared_str = line.split("\t")
            except ValueError:
                warn("Malformed line:\n{}".format(line), stacklevel = 0)
                continue
            unlabeled = int(unlabeled_id)
            if not (0 <= unlabeled <= 10000 * 10):
                error_string = "No such unlabeled node with id {}."
                warn(error_string.format(unlabeled_id), stacklevel = 0)
                continue
            try:
                shared_float = float(shared_str)
            except ValueError:
                error_string = "Error formatting value as float: {}."
                warn(error_string.format(shared_str), stacklevel = 0)
                continue
            lengths[unlabeled].append(shared_float)
    return lengths

dir1_files = set(listdir(args.dir1))
dir2_files = set(listdir(args.dir2))
files = dir2_files
if  dir1_files != dir2_files and not args.subset:
    print("Directories don't have same files")
    print("Difference is {}".format(dir1_files ^ dir2_files))
    exit()
else:
    files = dir1_files.intersection(dir2_files)
    assert len(files) > 0
    print("Using subset of size {}".format(len(files)))

total = 0
above = 0
below = 0
for filename in files:
    lengths_1 = {node_id: np.array(shared)
                 for node_id, shared
                 in file_to_lengths(join(args.dir1, filename)).items()}
    lengths_2 = {node_id: np.array(shared)
                 for node_id, shared
                 in file_to_lengths(join(args.dir2, filename)).items()}
    for node_id in lengths_1:
        statistic, p_value = stats.ks_2samp(lengths_1[node_id],
                                            lengths_2[node_id])
        if 0.95 < p_value:
            above += 1
        else:
            below += 1
        total += 1
    print("Current ratio above 0.95: {}".format(above / total))
