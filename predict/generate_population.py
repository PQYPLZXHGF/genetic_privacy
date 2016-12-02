#!/usr/bin/env python3

from argparse import ArgumentParser
from random import choice
from pickle import dump, HIGHEST_PROTOCOL

from population import HierarchicalIslandPopulation
from population_genomes import generate_genomes
from node import NodeGenerator
from recomb_genome import recombinators_from_directory, RecombGenomeGenerator
from island_model import tree_from_file
from sex import Sex

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def conditionalize_partner_probs(probabilities):
    conditional_probs = dict()
    for i, prob in enumerate(probabilities):
        greater_prob = sum(probabilities[i:])
        conditional_probs[i + 1] = probabilities[i] / greater_prob
    return conditional_probs

parser = ArgumentParser(description = "Generate a population and its associated genomes.")
parser.add_argument("tree_file",
                    help = "Describes hierarchical island model.")
parser.add_argument("recombination_dir",
                    help = "Directory containing Hapmap and decode data.")
parser.add_argument("--generation_size", help = "Number of individuals in each generation.",
                    type = int, default = 10000)
parser.add_argument("--num_generations", type = int, default = 10)
parser.add_argument("--no_genomes", action="store_true", default = False,
                    help = "Don't generate genomes for the individuals in the population.")
parser.add_argument("--non_paternity", "-p", type = float, default = 0,
                    help = "Rate with which the suspected father is not the true father.")
parser.add_argument("--adoption", "-a", type = float, default = 0,
                    help = "Rate with which the suspected mother and father are not the true mother and father.")
parser.add_argument("--multi_partner_prob", "-m", default = "1.0",
                    help = "Break down on number of partners people will have on average. Comma separated list of numbers between 0 and 1. First number the number of people who have 1 partner, next is 2 partners, etc. Numbers should sum up to 1.")
parser.add_argument("--output_file", default = "population.pickle",
                    help = "Outputs a pickle file containing a Population object to this file. This file will be clobbered if it exists.")

args = parser.parse_args()
if args.num_generations < 1:
    parser.error("num_generations must be >= 1")

if not 0 <= args.non_paternity <= 1:
    parser.error("Non-paternity rate must be in the range [0, 1]")
    
if not 0 <= args.adoption <= 1:
    parser.error("adoption rate must be in the range [0, 1]")

multi_partner_prob = [float(x) for x in args.multi_partner_prob.split(",")]
if not isclose(sum(multi_partner_prob), 1.0):
    parser.error("Multi partner probabilities must sum to 1. Summed to {}".format(sum(multi_partner_prob)))

cond_probs = conditionalize_partner_probs(multi_partner_prob)

node_generator = NodeGenerator()
founders = [node_generator.generate_node() for _ in range(args.generation_size)]

tree = tree_from_file(args.tree_file)
leaves = tree.leaves
for person in founders:
    tree.add_individual(choice(leaves), person)
population = HierarchicalIslandPopulation(tree)

for _ in range(args.num_generations - 1):
    population.new_generation(non_paternity_rate = args.non_paternity,
                              adoption_rate = args.adoption,
                              multi_partner_probs = cond_probs)

if not args.no_genomes:
    # tr = tracker.SummaryTracker()
    import time
    print("Loading recombination rates")
    start = time.perf_counter()
    recombinators = recombinators_from_directory(args.recombination_dir)
    stop = time.perf_counter()
    print("Recombination rates loaded. Took {} seconds".format(stop - start))
    chrom_sizes = recombinators[Sex.Male]._num_bases
    genome_generator = RecombGenomeGenerator(chrom_sizes)
    generate_genomes(population, genome_generator, recombinators, 3)
    # tr.print_diff()
    # summary.print_(summary.summarize(muppy.get_objects()))

# genomes = [m.genome for m in population.members]
# print("genome sizes: " + str(asizeof.asizeof(genomes) // 1024))
# print("population size: " + str(asizeof.asizeof(population) // 1024))

if args.output_file:
    with open(args.output_file, "wb") as pickle_file:
        # Trees cause deep recursion in the pickle module, so we need
        # to raise the recursion limit. This is the stack depth for
        # python functions, you may need to increase the native stack
        # depth using ulimit -s
        # https://docs.python.org/3.4/library/pickle.html#what-can-be-pickled-and-unpickled
        dump(population, pickle_file, protocol = HIGHEST_PROTOCOL)
