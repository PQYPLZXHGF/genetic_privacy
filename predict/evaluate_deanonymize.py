#!/usr/bin/env python3

from random import sample
from pickle import load
from argparse import ArgumentParser
from util import recent_common_ancestor, error_between_nodes

import pdb

from scipy import stats

from bayes_deanonymize import BayesDeanonymize
from population import PopulationUnpickler

parser = ArgumentParser(description = "Evaluate performance of classification.")
parser.add_argument("population")
parser.add_argument("classifier")
parser.add_argument("--num_node", "-n", type = int, default = 10)
parser.add_argument("--test_node", "-t", type = int, action = "append")
parser.add_argument("--subset_labeled", "-s", type = int, default = None,
                    help = "Chose a random subset of s nodes from the set of labeled nodes.")
args = parser.parse_args()


print("Loading population.")
with open(args.population, "rb") as pickle_file:
    population = PopulationUnpickler(pickle_file).load()

print("Loading classifier")
with open(args.classifier, "rb") as pickle_file:
    classifier = load(pickle_file)

nodes = set(member for member in population.members
             if member.genome is not None)
# nodes = set(member for member in population.generations[-1].members
#             if member.genome is not None)

if args.subset_labeled:
    classifier._labeled_nodes = sample(classifier._labeled_nodes,
                                       args.subset_labeled)

bayes = BayesDeanonymize(population, classifier)

id_mapping = population.id_mapping
labeled_nodes = set(id_mapping[node_id] for node_id
                    in classifier._labeled_nodes)
if args.test_node is not None and len(args.test_node) > 0:
    unlabeled = [id_mapping[node_id] for node_id in args.test_node]
else:
    unlabeled = sample(list(nodes - labeled_nodes),
                       args.num_node)

correct = 0
incorrect = 0
incorrect_examples = set()
incorrect_distances = []
no_common_ancestor = 0
print("Attempting to identify {} random nodes.".format(len(unlabeled)))
for i, node in enumerate(unlabeled):
    print("Iteration: {}, actual node ID: {}".format(i + 1, node._id))
    identified = bayes.identify(node.genome, node, population)
    assert len(identified) > 0
    # pdb.set_trace()
    if node in identified:
        correct += 1
        print("correct")
    else:
        incorrect_examples.add(node._id)
        print("incorrect")
        incorrect += 1
        import pdb
        for labeled_node in labeled_nodes:
            error = error_between_nodes(node, labeled_node,
                                        population.node_to_generation,
                                        False)
            if len(error[0]) > 0 or len(error[1]) > 0:
                pdb.set_trace()
        rca, distance = recent_common_ancestor(node, next(iter(identified)),
                                               population.node_to_generation)
        if distance is None:
            no_common_ancestor += 1
        else:
            incorrect_distances.append(distance)


print("{} correct, {} incorrect, {} total.".format(correct, incorrect,
                                                  len(unlabeled)))
print("{} percent accurate.".format(correct / len(unlabeled)))
print("Incorrectly guessed nodes: {}".format(incorrect_examples))
print("Relationship distance stats: {}".format(stats.describe(incorrect_distances)))
print("No common ancestor occured {} times.".format(no_common_ancestor))
