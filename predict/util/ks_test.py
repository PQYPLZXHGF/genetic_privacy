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
if  dir1_files != dir2_files:
    print("Directories don't have same files")
    print("Difference is {}".format(dir1_files ^ dir2_files))
    exit()

total = 0
above = 0
below = 0
for filename_1, filename_2 in zip(sorted(list(dir1_files)),
                                  sorted(list(dir2_files))):
    lengths_1 = {node_id: np.array(shared)
                 for node_id, shared
                 in file_to_lengths(join(args.dir1, filename_1)).items()}
    lengths_2 = {node_id: np.array(shared)
                 for node_id, shared
                 in file_to_lengths(join(args.dir2, filename_2)).items()}
    for node_id in lengths_1:
        statistic, p_value = stats.ks_2samp(lengths_1[node_id],
                                            lengths_2[node_id])
        if 0.95 < p_value:
            above += 1
        else:
            below += 1
        total += 1
    print("Current ratio above 0.95: {}".format(above / total))
