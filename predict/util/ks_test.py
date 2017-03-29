from argparse import ArgumentParser
from os import listdir
from os.path import join
from warnings import warn
from collections import defaultdict
from bisect import bisect_left
from pickle import dump
from math import isclose

import numpy as np
from scipy import stats

parser = ArgumentParser("Compare two simulation directories for discrepancies.")
parser.add_argument("dir1")
parser.add_argument("dir2")
parser.add_argument("--dir3")
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

def get_ks_for_files(file1, file2):
    lengths_1 = {node_id: np.array(shared)
                 for node_id, shared
                 in file_to_lengths(file1).items()}
    lengths_2 = {node_id: np.array(shared)
                 for node_id, shared
                 in file_to_lengths(file2).items()}
    stat_values = []
    for node_id in lengths_1:
        statistic, p_value = stats.ks_2samp(lengths_1[node_id],
                                            lengths_2[node_id])
        if isclose(p_value, 0.0):
            import pdb
            pdb.set_trace()
        stat_values.append((statistic, p_value))
    return stat_values

def get_all_ks(dir1, dir2, files):
    ks_data = []
    for filename in files:
        current_ks_data = get_ks_for_files(join(dir1, filename),
                                           join(dir2, filename))
        ks_data.extend(current_ks_data)
    return ks_data
        
dir1_files = set(listdir(args.dir1))
dir2_files = set(listdir(args.dir2))
if args.dir3:
    dir3_files = set(listdir(args.dir3))
else:
    dir3_files = None

files = dir2_files
if args.subset:
    files = dir1_files.intersection(dir2_files)
    if dir3_files is not None:
        files = files.intersection(dir3_files)
    assert len(files) > 0
    print("Using subset of size {}".format(len(files)))    
elif dir1_files != dir2_files:
    print("Directories 1 and 2 don't have same files")
    print("Difference is {}".format(dir1_files ^ dir2_files))
    exit()
elif dir3_files is not None and dir2_files != dir3_files:
    print("Directories 2 and 3 don't have same files")
    print("Difference is {}".format(dir2_files ^ dir3_files))
    exit()


if dir3_files is None:
    ks_data = get_all_ks(args.dir1, args.dir2, files)
    p_values = sorted([x[1] for x in ks_data])
    loc = bisect_left(p_values, 0.95)
    frac_above = (loc + 1) / len(p_values)
    print("{} of entries have p value of 0.95 or greater".format(frac_above))
else:
    dir1_dir2_ks = get_all_ks(args.dir1, args.dir2, files)
    dir2_dir3_ks = get_all_ks(args.dir2, args.dir3, files)
    with open("python_ks.pickle", "wb") as pickle_file:
        dump(dir1_dir2_ks, pickle_file)
    with open("rust_python_ks.pickle", "wb") as pickle_file:
        dump(dir2_dir3_ks, pickle_file)
    d1_d2_statistic = [x[0] for x in dir1_dir2_ks]
    d2_d3_statistic = [x[0] for x in dir2_dir3_ks]
    print(stats.ks_2samp(d1_d2_statistic, d2_d3_statistic))
