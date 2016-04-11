#!/usr/bin/env python3

from random import sample
from itertools import chain

from population import PopulationUnpickler
from sex import Sex
from classify_relationship import LengthClassifier
from recomb_genome import recombinators_from_directory, RecombGenomeGenerator

print("Loading population")
with open("population_1000.pickle", "rb") as pickle_file:
    population = PopulationUnpickler(pickle_file).load()

print("Loading recombination data.")
recombinators = recombinators_from_directory("../data/recombination_rates/")
chrom_sizes = recombinators[Sex.Male]._num_bases
genome_generator = RecombGenomeGenerator(chrom_sizes)

potentially_labeled = list(chain.from_iterable([generation.members
                                                for generation
                                                in population.generations[-3:]]))
labeled_nodes = sample(potentially_labeled, 10)
print("Populating length classifier.")
classifier = LengthClassifier(population, labeled_nodes, genome_generator,
                              recombinators)

# import pdb
# pdb.set_trace()