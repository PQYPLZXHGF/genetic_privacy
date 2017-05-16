#!/usr/bin/env python3

from collections import Counter, defaultdict
from datetime import datetime
from random import shuffle, getstate, setstate, seed
from pickle import load
from argparse import ArgumentParser
from math import sqrt
from sys import stdout

import pdb

from bayes_deanonymize import BayesDeanonymize
from population import PopulationUnpickler
from data_logging import (write_log, change_logfile_name,
                          stop_logging, start_logging)

parser = ArgumentParser(description = "Evaluate performance of classification.")
parser.add_argument("population")
parser.add_argument("classifier")
parser.add_argument("--data-logfile",
                    help = "Filename to write log data to.")
parser.add_argument("--num_node", "-n", type = int, default = 10)
parser.add_argument("--test_node", "-t", type = int, action = "append")
parser.add_argument("--subset_labeled", "-s", type = int, default = None,
                    help = "Chose a random subset of s nodes from the set of labeled nodes.")
parser.add_argument("--deterministic_random", "-d", action = "store_true",
                    help = "Seed the random number generator such that the same labeled nodes will be chosen on runs with the same number of nodes.")
parser.add_argument("--deterministic_labeled", "-ds", action = "store_true",
                    help = "Seed the random number generator to ensure labeled node subset is deterministic.")
args = parser.parse_args()

if args.data_logfile:
    change_logfile_name(args.data_logfile)
    start_logging()
else:
    stop_logging()

write_log("args", args)

print("Loading population.", flush = True)
with open(args.population, "rb") as pickle_file:
    population = PopulationUnpickler(pickle_file).load()

print("Loading classifier", flush = True)
with open(args.classifier, "rb") as pickle_file:
    classifier = load(pickle_file)

nodes = set(member for member in population.members
             if member.genome is not None)
# nodes = set(member for member in population.generations[-1].members
#             if member.genome is not None)

if args.subset_labeled:
    # we want the labeled nodes to be chosen randomly, but the same
    # random nodes chosen every time if the same number of labeled
    # nodes is chosen.
    sorted_labeled = list(classifier._labeled_nodes)
    sorted_labeled.sort()
    if args.deterministic_random or args.deterministic_labeled:
        rand_state = getstate()
        seed(42)
        shuffle(sorted_labeled)
        setstate(rand_state)
    else:
        shuffle(sorted_labeled)
    classifier._labeled_nodes = sorted_labeled[:args.subset_labeled]

write_log("labeled nodes", classifier._labeled_nodes)

bayes = BayesDeanonymize(population, classifier)

id_mapping = population.id_mapping
labeled_nodes = set(id_mapping[node_id] for node_id
                    in classifier._labeled_nodes)
if args.test_node is not None and len(args.test_node) > 0:
    unlabeled = [id_mapping[node_id] for node_id in args.test_node]
else:
    all_unlabeled = list(nodes - labeled_nodes)
    all_unlabeled.sort(key = lambda node: node._id)
    if args.deterministic_random:
        rand_state = getstate()
        seed(43)
        shuffle(all_unlabeled)
        setstate(rand_state)
    else:
        shuffle(all_unlabeled)
    unlabeled = all_unlabeled[:args.num_node]

write_log("to identify", [node._id for node in unlabeled])

correct = 0
incorrect = 0
# Maps generation -> counter with keys "correct" and "incorrect"
generation_error = defaultdict(Counter)
no_common_ancestor = 0
generation_map = population.node_to_generation
skipped = 0
# write_log("labeled_nodes", [node._id for node in labeled_nodes])
# write_log("target_nodes", [node._id for node in unlabeled])
print("Attempting to identify {} random nodes.".format(len(unlabeled)),
      flush = True)
write_log("start time", datetime.now())
for i, node in enumerate(unlabeled):
    print("Iteration: {}, actual node ID: {}".format(i + 1, node._id))
    # identified = bayes.identify(node.genome, node, population)
    identified, ln_ratio = bayes.identify(node.genome, node, population)
    # if ln_ratio < 0.1:
    #     skipped += 1
    #     continue
    assert len(identified) > 0
    # pdb.set_trace()
    node_generation = generation_map[node]
    if node in identified:
        generation_error[node_generation]["correct"] += 1
        correct += 1
        print("correct")
    else:
        generation_error[node_generation]["incorrect"] += 1
        print("incorrect")
        incorrect += 1
    write_log("evaluate", {"target node": node._id, "log ratio": ln_ratio,
                           "identified": set(x._id for x in identified)})
    stdout.flush()

write_log("end time", datetime.now())
print("{} skipped".format(skipped))
print("{} correct, {} incorrect, {} total.".format(correct, incorrect,
                                                   len(unlabeled)))
stdout.flush()

write_log("correct", correct)
write_log("incorrect", incorrect)
write_log("total", len(unlabeled))
total = correct + incorrect
percent_accurate = correct / total
std_dev = sqrt(percent_accurate * (1 - percent_accurate) * total) / total
print("{}±{:0.3} percent accurate.".format(percent_accurate, std_dev))
for generation, counter in generation_error.items():
    gen_correct = counter["correct"]
    gen_incorrect = counter["incorrect"]
    total = gen_correct + gen_incorrect
    format_string = "For generation {}: {} accuracy, {} total."
    print(format_string.format(generation, gen_correct / total, total))
