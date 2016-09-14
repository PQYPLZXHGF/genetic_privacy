#!/usr/bin/env python3

from argparse import ArgumentParser
from random import sample
from itertools import chain
from pickle import dump
from os import listdir

from population import PopulationUnpickler
from sex import Sex
from classify_relationship import generate_classifier, classifier_from_directory
from recomb_genome import recombinators_from_directory, RecombGenomeGenerator

parser = ArgumentParser(description = "Generate a classifier which can (hopefully) identify individuals in a population.")
parser.add_argument("population_file", help = "Pickled file with population")
parser.add_argument("work_dir",
                    help = "Directory to put shared length calculations in.")
parser.add_argument("num_iterations", type = int, default = 1000,
                    help = "Number of sample to collect from empirical distributions")
parser.add_argument("--recover", "-r", default = False,
                    action="store_true",
                    help = "work directory from interrupted run.")
parser.add_argument("--gen_back", "-g", type = int, default = 7,
                    help = "Ignore common ancestry more than the given number of generations back.")
parser.add_argument("--num_labeled_nodes", "-n", type = int, default = 0)
parser.add_argument("--output_pickle", default = "classifier.pickle")

args = parser.parse_args()

print("Loading population")
with open(args.population_file, "rb") as pickle_file:
    population = PopulationUnpickler(pickle_file).load()

print("Loading recombination data.")
recombinators = recombinators_from_directory("../data/recombination_rates/")
chrom_sizes = recombinators[Sex.Male]._num_bases
genome_generator = RecombGenomeGenerator(chrom_sizes)

if not args.recover:
    potentially_labeled = list(chain.from_iterable([generation.members
                                                    for generation
                                                    in population.generations[-3:]]))
    if args.num_labeled_nodes <= 0:
        num_labeled_nodes = population.size // 100
    else:
        num_labeled_nodes = args.num_labeled_nodes
    labeled_nodes = sample(potentially_labeled, num_labeled_nodes)
else:
    print("Recovering run")
    labeled_nodes = [population.id_mapping[int(filename)]
                     for filename in listdir(args.work_dir)]
print("Populating length classifier.")

clobber = not args.recover && num_iterations != 0

classifier = generate_classifier(population, labeled_nodes,
                                 genome_generator, recombinators,
                                 args.work_dir,
                                 iterations = args.num_iterations,
                                 clobber = clobber,
                                 generations_back_shared = args.gen_back)

print("Pickling classifier")
with open(args.output_pickle, "wb") as pickle_file:
    dump(classifier, pickle_file)

# import pdb
# pdb.set_trace()
