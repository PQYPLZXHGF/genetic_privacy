from collections import namedtuple, defaultdict
from itertools import chain, product, combinations
from os import listdir, makedirs
from os.path import join, exists
from random import sample, random
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
        if (query_node, labeled_node) not in self._distributions:
            if shared_length == 0:
                return self._cryptic_distribution.zero_prob
            ret = 1 - self._empirical_cryptic_distribution(shared_length)
            if ret == 0:
                return 1 - self._empirical_cryptic_distribution.y[-2]
            return ret
        shape, scale, zero_prob = self._distributions[query_node, labeled_node]
        if shared_length == 0:
            return zero_prob
        ret = gamma.cdf(shared_length, a = shape,
                        scale = scale)
        if ret > 0.5:
            ret = 1 - ret
        ret = ret * 2 * (1 - zero_prob)
        if ret <= 0.0:
            return ZERO_REPLACE
        return ret

    def get_batch_smoothing(self, lengths):
        lengths = np.asarray(lengths, dtype = np.uint64)
        probs = np.empty_like(lengths, dtype = np.float64)
        probs[lengths >= 30000000] = 0.005
        probs[lengths < 30000000] = 0.03
        probs[lengths == 0] = 1
        return probs

    def get_batch_cryptic_ecdf(self, lengths):
        """
        Get probabilities for lengths based on ECDF of lengths of
        unrelated pairs.
        """
        len_zero_prob = self._cryptic_distribution.zero_prob
        lengths = np.asarray(lengths, dtype = np.uint32) 
        zero_len_i = (lengths == 0)
        nonzero_len_i = np.invert(zero_len_i)
        
        # ret = (1 - self._empirical_cryptic_distribution(lengths)) * (1 - len_zero_prob)
        # ret = 1 - self._empirical_cryptic_distribution(lengths)
        ret = 1 - self._empirical_cryptic_distribution_nozero(lengths) * (1 - len_zero_prob)
        zero_i = (ret == 0)
        min_val = 1 - self._empirical_cryptic_distribution.y[-2]
        # min_val = 0.00000005
        # ret[ret < min_val] = min_val
        ret[zero_i] = min_val

        # ret[zero_len_i] = self._cryptic_distribution.zero_prob
        ret[zero_len_i] = len_zero_prob
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

    def get_batch_ecdf(self, lengths, labeled_nodes):
        lengths = np.asarray(lengths, dtype = np.uint32)
        labeled_nodes = np.asarray(labeled_nodes, dtype = np.uint32)
        sort_i = np.argsort(labeled_nodes)
        labeled_nodes = labeled_nodes[sort_i]
        lengths = lengths[sort_i]
        unique_labeled = np.unique(labeled_nodes)
        right_i = np.searchsorted(labeled_nodes, unique_labeled, side = "right")
        partitioned_lengths = np.split(lengths, right_i)
        ecdf_map = self._cryptic_ecdf
        probabilities = []
        for node_i, labeled_node in enumerate(unique_labeled.tolist()):
            sub_lengths = partitioned_lengths[node_i]
            temp_probs = np.empty_like(sub_lengths, dtype = np.float64)
            zero_i = (sub_lengths == 0)
            nonzero_i = np.invert(zero_i)
            frac_zero, ecdf = ecdf_map[labeled_node]
            temp_probs[nonzero_i] = (1 - frac_zero) * (1 - ecdf(sub_lengths[nonzero_i]))
            temp_probs[zero_i] = frac_zero
            # temp_probs = 1 - ecdf_map[labeled_node](lengths)
            probabilities.append(temp_probs)
        ret = np.concatenate(probabilities)

        # inverse initial argsort so that probabilities are returned
        # in the order given
        # see https://arogozhnikov.github.io/2015/09/29/NumpyTipsAndTricks1.html#Even-faster-inverse-permutation
        inverse_i = np.empty(len(sort_i), dtype = np.uint32)
        inverse_i[sort_i] = np.arange(len(sort_i))
        
        return ret[inverse_i]
        
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
            if (unlabeled != labeled and
                len(ancestors[unlabeled].intersection(ancestors[labeled])) != 0)]


# At some point this should probably be turned into a "builder" class,
# too much state is getting passed along in this long parameter list.
def generate_classifier(population, labeled_nodes, genome_generator,
                        recombinators, directory, clobber = True,
                        iterations = 1000, generations_back_shared = 7,
                        min_segment_length = 0, non_paternity = 0.0):    
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
                            min_segment_length = min_segment_length,
                            generations_back_shared = generations_back_shared,
                            non_paternity = non_paternity)
    print("Generating cryptic relative parameters")
    population.clean_genomes()
    generate_genomes(population, genome_generator, recombinators, 3,
                     true_genealogy = True)
    cryptic_lens = cryptic_lengths(population, labeled_nodes,
                                   generations_back_shared,
                                   min_segment_length)
    # cryptic_ecdf = labeled_cryptic_ecdf(population, labeled_nodes,
    #                                     generations_back_shared,
    #                                     min_segment_length)
    # classifier._cryptic_ecdf = cryptic_ecfd
    # We still fit a hurdle gamma to get the zero probability, and to
    # keep the option of switching back in the future easier.
    cryptic_params = HurdleGammaParams(*fit_hurdle_gamma(cryptic_lens))
    print("Generating classifiers.")
    classifier = classifier_from_directory(directory, population.id_mapping)
    classifier._cryptic_distribution = cryptic_params
    print("Generating ecdf")
    classifier._empirical_cryptic_distribution = ECDF(cryptic_lens, "left")
    nonzero_cryptic_lens = cryptic_lens[np.nonzero(cryptic_lens)]
    classifier._empirical_cryptic_distribution_nozero = ECDF(nonzero_cryptic_lens,
                                                             "left")

    return classifier

def labeled_cryptic_ecdf(population, labeled_nodes, generations_back_shared,
                            min_segment_length = 0):
    unlabeled = [node for node in population.members
                 if node.genome is not None]
    unlabeled = set(sample(unlabeled, len(unlabeled) // 2))
    related = related_pairs(unlabeled, labeled_nodes, population,
                            generations_back_shared)
    related_map = defaultdict(set)
    for node_a, node_b in related:
        related_map[node_a].add(node_b)
        related_map[node_b].add(node_a)
        
    cryptic_ecdf = dict()    
    for labeled_node in labeled_nodes:
        labeled_related = related_map[labeled_node]
        unrelated_pairs = ((unlabeled_node, labeled_node)
                           for unlabeled_node in unlabeled
                           if unlabeled_node not in labeled_related)
        lengths_iter = (shared_segment_length_genomes(node_a.genome,
                                                      node_b.genome,
                                                      min_segment_length)
                        for node_a, node_b in unrelated_pairs)
        lengths = np.fromiter(lengths_iter, dtype = np.uint32)
        nonzero_i = (lengths != 0)
        frac_zero = (len(lengths) - np.sum(nonzero_i)) / len(lengths)
        ecdf = ECDF(lengths[nonzero_i], "left")
        cryptic_ecdf[labeled_node._id] = (frac_zero, ecdf)
    return cryptic_ecdf
                
    

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
                        generations_back_shared = 7,
                        non_paternity = 0.0):

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
    suppressor = ParentSuppressor(non_paternity, 0.0)
    print("Calculating shared lengths.")
    for i in range(iterations):
        print("iteration {}".format(i))
        print("Cleaning genomes.")
        population.clean_genomes()
        print("Perturbing parentage")
        suppressor.suppress(population)
        print("Generating genomes")
        generate_genomes(population, genome_generator, recombinators, 3,
                         true_genealogy = False)
        print("Calculating shared length")
        _calculate_shared_to_fds(pairs, fds, min_segment_length)
        print("Fixing perturbation")
        suppressor.unsuppress()
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

SuppressedParents = namedtuple("SuppressedParents", ["father", "mother"])
class ParentSuppressor:
    """
    Supress the suspected father and mother of a node.
    This is to simulate false paternity and adoption.
    """
    def __init__(self, father_percent = 0.0, mother_percent = 0.0):
        self._father_percent = float(father_percent)
        self._mother_percent = float(mother_percent)
        self._changed_nodes = dict()

    def suppress(self, population):
        father = self._father_percent
        mother = self._mother_percent
        if father == 0.0 and mother == 0.0:
            return
        generations = (generation.members for generation in
                       population.generations[1:])
        for node in chain.from_iterable(generations):
            suppressed_father = None
            suppressed_mother = None
            if node.suspected_father is not None and random() < father:
                suppressed_father = node.suspected_father
                node.suspected_father = None
                node._suspected_father_id = None
                pass
            if node.suspected_mother is not None and random() < mother:
                suppressed_mother = node.suspected_mother
                node.suspected_mother = None
                node._suspected_mother_id = None
            if suppressed_father is not None or suppressed_mother is not None:
                self._changed_nodes[node] = SuppressedParents(suppressed_father,
                                                              suppressed_mother)
                

    def unsuppress(self):
        for node, parents in self._changed_nodes.items():
            father, mother = parents
            if father is not None:
                node.suspected_father = father
                node._suspected_father_id = father._id
            if mother is not None:
                node.suspected_mother = mother
                node._suspected_mother_id = mother._id
        self._changed_nodes = dict()
