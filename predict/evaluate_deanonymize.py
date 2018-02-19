#!/usr/bin/env python3

from random import shuffle, getstate, setstate, seed
from pickle import load, dump
from argparse import ArgumentParser
from os.path import exists

import pdb

from evaluation import Evaluation
from expansion import ExpansionData
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
                    help = "Chose a random subset of s nodes from the set of labeled nodes. If using expansion rounds, this is the size of the initial set of labeled nodes.")
parser.add_argument("--ibd-threshold", type = int, default = 5000000,
                    help = "IBD segments smaller than this value will "
                    "go undetected")
parser.add_argument("--deterministic_random", "-d", action = "store_true",
                    help = "Seed the random number generator such that the same labeled nodes will be chosen on runs with the same number of nodes.")
parser.add_argument("--deterministic_labeled", "-ds", action = "store_true",
                    help = "Seed the random number generator to ensure labeled node subset is deterministic.")
# parser.add_argument("--expansion-round", type = int, default = 1)
parser.add_argument("--search-related", type = int, default = False,
                    help = "Search only nodes that are related to labeled nodes for which there is nonzero ibd.")
parser.add_argument("--expansion-rounds-data",
                    help = "Pickle file with data from expansion rounds.")

args = parser.parse_args()

if args.expansion_rounds_data:
    expansion_file_exists = exists(args.expansion_rounds_data)
    if not expansion_file_exists and args.subset_labeled is None:
        parser.error("A subset of labeled nodes is necessary for expansion rounds when expansion rounds data file does not already exist.")
    if expansion_file_exists:
        with open(args.expansion_rounds_data, "rb") as expansion_file:
            expansion_data = load(expansion_file)
    else:
        expansion_data = None


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

# nodes = set(member for member in population.generations[-1].members
#             if member.genome is not None)

evaluation = Evaluation(population, classifier,
                        ibd_threshold = args.ibd_threshold,
                        search_related = args.search_related)
original_labeled = set(evaluation.labeled_nodes)
if args.expansion_rounds_data and expansion_data is not None:
    evaluation.labeled_nodes = expansion_data.labeled_nodes
    expansion_data.adjust_genomes(population)
elif args.subset_labeled:
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
    
    evaluation.labeled_nodes = sorted_labeled[:args.subset_labeled]

if args.expansion_rounds_data and expansion_data is None:
    expansion_data = ExpansionData(evaluation.labeled_nodes)

write_log("labeled nodes", evaluation.labeled_nodes)

id_mapping = population.id_mapping
nodes = set(member for member in population.members
             if member.genome is not None)
labeled_nodes = set(id_mapping[node_id] for node_id
                    in evaluation.labeled_nodes)
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

if not args.expansion_rounds_data:
    evaluation.run_evaluation(unlabeled)
    evaluation.print_metrics()
else:
    identify_candidates = set(id_mapping[node] for node
                              in original_labeled - set(evaluation.labeled_nodes))
    added = evaluation.run_expansion_round(identify_candidates,
                                           expansion_data,
                                           args.expansion_rounds_data)
    expansion_data.add_round(added)
    with open(args.expansion_rounds_data, "wb") as expansion_file:
        dump(expansion_data, expansion_file)
