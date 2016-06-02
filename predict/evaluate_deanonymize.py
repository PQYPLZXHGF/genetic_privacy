#!/usr/bin/env python3

from random import sample
from pickle import load

import pdb

from bayes_deanonymize import BayesDeanonymize
from population import PopulationUnpickler

NUM_LABELED = 400

print("Loading population.")
with open("population_10000.pickle", "rb") as pickle_file:
    population = PopulationUnpickler(pickle_file).load()

print("Loading classifier")
with open("classifier_2000_gen6.pickle", "rb") as pickle_file:
    classifier = load(pickle_file)

print("Fixing persistence")
classifier.fix_persistence(population)
print("Checking labeled nodes")
all_nodes = set(population.members)
for node in classifier._labeled_nodes:
    assert node in all_nodes

last_generation = population.generations[-1].members

bayes = BayesDeanonymize(population, classifier)

unlabeled = sample(list(set(last_generation) - set(classifier._labeled_nodes)),
                   200)
# unlabeled = [choice(list(set(last_generation) - labeled_nodes))]
correct = 0
incorrect = 0
print("Attempting to identify {} random nodes.".format(len(unlabeled)))
i = 0
for node in unlabeled:
    print(i)
    identified = bayes.identify(node.genome, node)
    # pdb.set_trace()
    if node in identified:
        correct += 1
        print("correct")
    else:
        print("incorrect")
        incorrect += 1
    i += 1


print("{} correct, {} incorrect, {} total.".format(correct, incorrect,
                                                  len(unlabeled)))
print("{} percent accurate.".format(correct / len(unlabeled)))
