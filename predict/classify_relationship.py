from collections import namedtuple, defaultdict
from itertools import chain, product, combinations
from os import listdir, makedirs
from os.path import join, exists
from random import sample
from shutil import rmtree
from warnings import warn
import pdb

from scipy.stats import gamma
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
# import pyximport; pyximport.install()

from common_segments import common_segment_lengths
from population_genomes import generate_genomes
from population_statistics import ancestors_of
from gamma import fit_hurdle_gamma

# ZERO_REPLACE = 1e-20
ZERO_REPLACE = 0.03

GammaParams = namedtuple("GammaParams", ["shape", "scale"])
HurdleGammaParams = namedtuple("HurdleGammaParams", ["shape", "scale", "zero_prob"])

class LengthClassifier:
    """
    Classifies based total length of shared segments
    """
    def __init__(self, distributions, labeled_nodes,
                 cryptic_distribution = None,
                 empirical_cryptic_lengths = None):
        self._distributions = distributions
        self._labeled_nodes = labeled_nodes
        self._cryptic_distribution = cryptic_distribution
        if empirical_cryptic_lengths is not None:
            self._empirical_cryptic_distribution = ECDF(empirical_cryptic_lengths, "left")
        else:
            self._empirical_cryptic_distribution = None
            
    def get_probability(self, shared_length, query_node, labeled_node):
        """
        Returns the probability that query_node and labeled_node have total
        shared segment length shared_length
        """
        shape, scale, zero_prob =  self._distributions[query_node, labeled_node]
        if shared_length == 0:
            # return 1 - zero_prob
            return zero_prob
        # return (1 - zero_prob) * gamma.pdf(shared_length, a = shape,
        #                                    scale = scale) * GAMMA_SCALE
        # return 1 - gamma.cdf(shared_length, a = shape,
        #                      scale = scale)
        ret = gamma.cdf(shared_length, a = shape,
                        scale = scale)
        if ret > 0.5:
            ret = 1 - ret
        ret = ret * 2 * (1 - zero_prob)
        if ret <= 0.0:
            return ZERO_REPLACE
        return ret

    def get_batch_cryptic_ecdf(self, lengths):
        """
        Get probabilities for lengths based on ECDF of lengths of
        unrelated pairs.
        """
        lengths = np.asarray(lengths, dtype = np.uint32) 
        zero_len_i = (lengths == 0)
        nonzero_len_i = np.invert(zero_len_i)
        
        ret = 1 - self._empirical_cryptic_distribution(lengths)
        
        zero_i = (ret == 0)
        min_val = np.min(ret[np.invert(zero_i)])
        ret[zero_i] = min_val

        ret[zero_len_i] = self._cryptic_distribution.zero_prob
        return ret

    def get_batch_cryptic(self, lengths):
        """
        Get probabilities for lengths based on a hurdle gamma fit of
        unrelated pairs.
        """
        assert self._cryptic_distribution is not None
        shape, scale, zero_prob = self._cryptic_distribution
        lengths = np.array(lengths, dtype = np.uint32)
        zero_i = (lengths == 0)
        nonzero_i = np.invert(zero_i)
        ret = np.empty_like(lengths, dtype = np.float64)
        ret[zero_i] = zero_prob
        
        gamma_probs = gamma.cdf(lengths[nonzero_i], a = shape, scale = scale)
        greater_i = gamma_probs > 0.5
        gamma_probs[greater_i] = 1 - gamma_probs[greater_i]
        gamma_probs = gamma_probs * 2 * (1 - zero_prob)
        ret[nonzero_i] = gamma_probs
        ret[ret <= 0.0] = ZERO_REPLACE
        return ret

        
    def get_batch_probability(self, lengths, query_nodes, labeled_nodes):
        lengths = np.array(lengths, dtype = np.uint32)
        zero_i = (lengths == 0)
        nonzero_i = np.invert(zero_i)
        distributions = self._distributions
        params = (distributions[query_node, labeled_node]
                  for query_node, labeled_node
                  in zip(query_nodes, labeled_nodes))
        shape_scale_zero = list(zip(*params))
        shapes = np.array(shape_scale_zero[0], dtype = np.float64)
        scales = np.array(shape_scale_zero[1], dtype = np.float64)
        zero_prob = np.array(shape_scale_zero[2], dtype = np.float64)
        del shape_scale_zero
        ret = np.empty_like(lengths, dtype = np.float64)
        # ret[zero_i] = 1 - zero_prob[zero_i]
        ret[zero_i] = zero_prob[zero_i]
        # ret[zero_i] = 1.0
        gamma_probs = gamma.cdf(lengths[nonzero_i],
                                a = shapes[nonzero_i],
                                scale = scales[nonzero_i])
        # gamma_probs = np.ones_like(lengths, dtype = np.float64)

        # gamma_probs[gamma_probs == 0.0] = ZERO_REPLACE

        # gamma_probs = np.exp(np.log(gamma_probs) + np.log(1 - zero_prob[nonzero_i]))
        # gamma_probs = (1 - gamma_probs)
        greater_i = gamma_probs > 0.5
        gamma_probs[greater_i] = 1 - gamma_probs[greater_i]
        gamma_probs = gamma_probs * 2 * (1 - zero_prob[nonzero_i])
        ret[nonzero_i] = gamma_probs
        ret[ret <= 0.0] = ZERO_REPLACE
        # ret[ret > 1.0] = 1.0
        return ret

    def __contains__(self, item):
        return item in self._distributions

def related_pairs(unlabeled_nodes, labeled_nodes, population, generations):
    """
    Given a population and labeled nodes, returns a list of pairs of nodes
    (unlabeled node, labeled node)
    where the labeled node and unlabeled node share at least 1 common ancestor
    going back generation generations from the latest generation.
    """
    generation_map = population.node_to_generation
    num_generations = population.num_generations
    if type(labeled_nodes) != set:
        labeled_nodes = set(labeled_nodes)
    ancestors = dict()
    for node in chain(unlabeled_nodes, labeled_nodes):
        node_generation = generation_map[node]
        from_latest = (num_generations - node_generation - 1)
        generations_back =  generations - from_latest
        ancestors[node] = ancestors_of(node, generations_back)
    return [(unlabeled, labeled) for unlabeled, labeled
            in product(unlabeled_nodes, labeled_nodes)
            if len(ancestors[unlabeled].intersection(ancestors[labeled])) != 0]


def generate_classifier(population, labeled_nodes, genome_generator,
                        recombinators, directory, clobber = True,
                        iterations = 1000, generations_back_shared = 7):    
    if not exists(directory):
        makedirs(directory)
    elif clobber:
        rmtree(directory)
        makedirs(directory)
    num_generations = population.num_generations
    clear_index = num_generations - generations_back_shared - 1
    to_clear = population.generations[clear_index].members
    for node in to_clear:
        node.suspected_mother = None
        node.suspected_mother_id = None
        node.suspected_father = None
        node.suspected_father_id = None
    if 0 < iterations:
        shared_to_directory(population, labeled_nodes, genome_generator,
                            recombinators, directory, clobber = clobber,
                            iterations = iterations,
                            generations_back_shared = generations_back_shared)
    print("Generating cryptic relative parameters")
    population.clean_genomes()
    generate_genomes(population, genome_generator, recombinators, 3,
                     true_genealogy = True)
    cryptic_lens = cryptic_lengths(population, labeled_nodes,
                                   generations_back_shared)
    # We still fit a hurdle gamma to get the zero probability, and to
    # keep the option of switching back in the future easier.
    cryptic_params = HurdleGammaParams(*fit_hurdle_gamma(cryptic_lens))
    print("Generating classifiers.")
    classifier = classifier_from_directory(directory, population.id_mapping)
    classifier._cryptic_distribution = cryptic_params
    print("Generating ecdf")
    classifier._empirical_cryptic_distribution = ECDF(cryptic_lens, "left")
    return classifier

def cryptic_lengths(population, labeled_nodes, generations_back_shared,
                    min_segment_length = 0):
    nodes = sample(population.generations[-1].members, 1000)
    # related_labeled = related_pairs(labeled_nodes, labeled_nodes, population,
    #                                 generations_back_shared)
    related_labeled = related_pairs(nodes, nodes, population,
                                    generations_back_shared)
    related_map = defaultdict(set)
    for node_a, node_b in related_labeled:
        related_map[node_a].add(node_b)
        related_map[node_b].add(node_a)

    # labeled_pairs = combinations(labeled_nodes, 2)
    labeled_pairs = combinations(nodes, 2)
    unrelated_pairs = ((a, b) for a, b in labeled_pairs
                       if a not in related_map[b])
    shared_iter = (shared_segment_length_genomes(node_a.genome,
                                                 node_b.genome,
                                                 min_segment_length)
                   for node_a, node_b in unrelated_pairs)
    return np.fromiter(shared_iter, dtype = np.uint32)

def shared_to_directory(population, labeled_nodes, genome_generator,
                        recombinators, directory, min_segment_length = 0,
                        clobber = True, iterations = 1000,
                        generations_back_shared = 7):

    labeled_nodes = set(labeled_nodes)
    unlabeled_nodes = chain.from_iterable(generation.members
                                          for generation
                                          in population.generations[-3:])
    unlabeled_nodes = set(unlabeled_nodes) - labeled_nodes
    print("Finding related pairs.")
    pairs = related_pairs(unlabeled_nodes, labeled_nodes, population,
                          generations_back_shared)
    print("{} related pairs.".format(len(pairs)))
    print("Opening file descriptors.")
    if clobber:
        mode = "w"
    else:
        mode = "a"
    fds = {node: open(join(directory, str(node._id)), mode)
           for node in labeled_nodes}
    print("Calculating shared lengths.")
    for i in range(iterations):
        print("iteration {}".format(i))
        print("Cleaning genomes.")
        population.clean_genomes()
        print("Generating genomes")
        generate_genomes(population, genome_generator, recombinators, 3,
                         true_genealogy = False)
        print("Calculating shared length")
        _calculate_shared_to_fds(pairs, fds, min_segment_length)
    for fd in fds.values():
        fd.close()
        
def _calculate_shared_to_fds(pairs, fds, min_segment_length):
    """
    Calculate the shared length between the pairs, and store the
    shared length in the given directory. Each labeled node has a file
    in the given directory. The files contain tab separated entries,
    where the first entry is the unlabeled node id, and the second
    entry is the amount of shared material.
    """
    shared_iter = ((unlabeled, labeled,
                    shared_segment_length_genomes(unlabeled.genome,
                                                  labeled.genome,
                                                  min_segment_length))
                   for unlabeled, labeled in pairs)
    for unlabeled, labeled, shared in shared_iter:
        fd = fds[labeled]
        fd.write("{}\t{}\n".format(unlabeled._id, shared))

def classifier_from_directory(directory, id_mapping):
    distributions = distributions_from_directory(directory, id_mapping)
    labeled_nodes = set(int(filename) for filename in listdir(directory))
    return LengthClassifier(distributions, labeled_nodes)

def distributions_from_directory(directory, id_mapping):
    """
    Calculate distributions from a directory created by
    calculate_shared_to_directory.
    """
    distributions = dict()
    for labeled_filename in listdir(directory):
        lengths = defaultdict(list)
        labeled = int(labeled_filename)
        with open(join(directory, labeled_filename), "r") as labeled_file:
            for line in labeled_file:
                # If the program crashed, the output can be left in an
                # inconsistent state.
                try:
                    unlabeled_id, shared_str = line.split("\t")
                except ValueError:
                    warn("Malformed line:\n{}".format(line), stacklevel = 0)
                    continue
                unlabeled = int(unlabeled_id)
                if unlabeled not in id_mapping:
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
        for unlabeled, lengths in lengths.items():
            shape, scale, zero_prob = fit_hurdle_gamma(np.array(lengths,
                                                                dtype = np.uint32))
            if shape is None:
                continue
            params = HurdleGammaParams(shape, scale, zero_prob)
            distributions[unlabeled, labeled] = params
    return distributions
    
def shared_segment_length_genomes(genome_a, genome_b, minimum_length):
    lengths = common_segment_lengths(genome_a, genome_b)
    seg_lengths = (x for x in lengths if x >= minimum_length)
    return sum(seg_lengths)
    
def _shared_segment_length(node_a, node_b, minimum_length):
    return shared_segment_length_genomes(node_a.genome, node_b.genome,
                                          minimum_length)
