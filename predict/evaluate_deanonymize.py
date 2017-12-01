#!/usr/bin/env python3

from collections import Counter, defaultdict, namedtuple
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
                    help = "Chose a random subset of s nodes from the set of labeled nodes. If using expansion rounds, this is the size of the initial set of labeled nodes.")
parser.add_argument("--ibd-threshold", type = int, default = 5000000,
                    help = "IBD segments smaller than this value will "
                    "go undetected")
parser.add_argument("--deterministic_random", "-d", action = "store_true",
                    help = "Seed the random number generator such that the same labeled nodes will be chosen on runs with the same number of nodes.")
parser.add_argument("--deterministic_labeled", "-ds", action = "store_true",
                    help = "Seed the random number generator to ensure labeled node subset is deterministic.")
parser.add_argument("--expansion-rounds", type = int, default = 1)

args = parser.parse_args()

if args.expansion_rounds > 1 and args.subset_labeled is None:
    parser.error("A subset of labeled nodes is necessary for expansion rounds.")
if args.data_logfile:
    change_logfile_name(args.data_logfile)
    start_logging()
else:
    stop_logging()

write_log("args", args)

IdentifyResult = namedtuple("IdentifyResult", ["target_node",
                                               "identified_node",
                                               "ln_ratio",
                                               "correct",
                                               "run_number"])

class Evaluation:
    def __init__(self, population, classifier, labeled_nodes = None,
                 ibd_threshold = 0):
        self._population = population
        self._classifier = classifier
        if labeled_nodes is not None:
            self.set_labeled_nodes(labeled_nodes)
        self._bayes = BayesDeanonymize(population, classifier)
        self._run_number = 0
        self._ibd_threshold = ibd_threshold
        self.reset_metrics()

    @property
    def accuracy(self):
        total = self.correct + self.incorrect
        return self.correct / total

    @property
    def labeled_nodes(self):
        return self._classifier._labeled_nodes

    @labeled_nodes.setter
    def labeled_nodes(self, labeled_nodes):
        self._classifier._labeled_nodes = labeled_nodes

    def print_metrics(self):
        print("{} correct, {} incorrect, {} total.".format(self.correct,
                                                           self.incorrect,
                                                           len(unlabeled)))
        stdout.flush()

        write_log("correct", self.correct)
        write_log("incorrect", self.incorrect)
        write_log("total", len(unlabeled))
        total = self.correct + self.incorrect
        percent_accurate = self.accuracy
        std_dev = sqrt(percent_accurate * (1 - percent_accurate) * total) / total
        print("{}±{:0.3} percent accurate.".format(percent_accurate, std_dev))
        for generation, counter in self.generation_error.items():
            gen_correct = counter["correct"]
            gen_incorrect = counter["incorrect"]
            total = gen_correct + gen_incorrect
            format_string = "For generation {}: {} accuracy, {} total."
            print(format_string.format(generation, gen_correct / total, total))

    def _evaluate_node(self, node):
        identified, ln_ratio = self._bayes.identify(node.genome, node,
                                                    self._ibd_threshold)
        assert len(identified) > 0
        node_generation = self._population.node_to_generation[node]
        if node in identified:
            self.generation_error[node_generation]["correct"] += 1
            self.correct += 1
            print("correct")
        else:
            self.generation_error[node_generation]["incorrect"] += 1
            print("incorrect")
            self.incorrect += 1
            
        write_log("evaluate", {"target node": node._id,
                               "log ratio": ln_ratio,
                               "identified": set(x._id for x in identified),
                               "run_number": self._run_number})
        stdout.flush()
        return IdentifyResult(node, identified,
                              ln_ratio,
                              node in identified,
                              self._run_number)

    def reset_metrics(self):
        # Maps generation -> counter with keys "correct" and "incorrect"
        self.generation_error = defaultdict(Counter)
        self.identify_results = []
        self.correct = 0
        self.incorrect = 0

    def run_evaluation(self, unlabeled):
        # generation_map = population.node_to_generation
        # write_log("labeled_nodes", [node._id for node in labeled_nodes])
        # write_log("target_nodes", [node._id for node in unlabeled])
        print("Attempting to identify {} random nodes.".format(len(unlabeled)),
              flush = True)
        write_log("start time", datetime.now())
        for i, node in enumerate(unlabeled):
            print("Iteration: {}, actual node ID: {}".format(i + 1, node._id))
            self.identify_results.append(self._evaluate_node(node))

        write_log("end time", datetime.now())
        self._run_number += 1
        return self.identify_results


print("Loading population.", flush = True)
with open(args.population, "rb") as pickle_file:
    population = PopulationUnpickler(pickle_file).load()

print("Loading classifier", flush = True)
with open(args.classifier, "rb") as pickle_file:
    classifier = load(pickle_file)


# nodes = set(member for member in population.generations[-1].members
#             if member.genome is not None)

evaluation = Evaluation(population, classifier,
                        ibd_threshold = args.ibd_threshold)
original_labeled = set(evaluation.labeled_nodes)
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
    
    evaluation.labeled_nodes = sorted_labeled[:args.subset_labeled]

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

if args.expansion_rounds <= 1:
    evaluation.run_evaluation(unlabeled)
    evaluation.print_metrics()
else:
    total_added = 0
    identify_candidates = set(id_mapping[node] for node
                              in original_labeled - set(evaluation.labeled_nodes))
    for i in range(args.expansion_rounds):
        print("On expansion round {}".format(i))
        to_evaluate = list(identify_candidates)
        round_added = 0
        for i, node in enumerate(to_evaluate):
            evaluation.run_evaluation([node])
            result = evaluation.identify_results[-1]
            if result.correct and result.ln_ratio > 10:
                evaluation.labeled_nodes.append(result.target_node._id)
                identify_candidates.remove(result.target_node)
                round_added += 1
            if i % 20 == 0:
                evaluation.print_metrics()
        total_added += round_added
        write_log("expansion_round", {"round": i, "added": round_added,
                                      "accuracy": evaluation.accuracy})
        if round_added == 0:
            print("No nodes added this round. Ceasing after {} iterations.".format(i + 1))
            break
        print("Added {} nodes this round.".format(round_added))
    print("{} total nodes added to the labeled set.")
    evaluation.print_metrics()
