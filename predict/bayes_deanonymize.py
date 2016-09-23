from math import isnan

# import pyximport; pyximport.install()
import numpy as np

from classify_relationship import (LengthClassifier,
                                   shared_segment_length_genomes)
MINIMUM_LABELED_NODES = 5
INF = float("inf")
INF_REPLACE = 1.0
ZERO_REPLACE = 1e-20

class BayesDeanonymize:
    def __init__(self, population, classifier = None):
        self._population = population
        if classifier is None:
            self._length_classifier = LengthClassifier(population, 1000)
        else:
            self._length_classifier = classifier

    def _compare_genome_node(self, node, genome, cache):
        probabilities = []
        batch_node_id = []
        batch_labeled_node_id = []
        batch_lengths = []
        length_classifier = self._length_classifier
        id_map = self._population.id_mapping
        for labeled_node_id in length_classifier._labeled_nodes:
            labeled_node = id_map[labeled_node_id]
            if labeled_node in cache:
                shared = cache[labeled_node]
            else:
                shared = shared_segment_length_genomes(genome,
                                                       labeled_node.genome,
                                                       0)
                cache[labeled_node] = shared

            if (node._id, labeled_node._id) not in length_classifier:
                if shared == 0:
                    probabilities.append(INF_REPLACE)
                else:
                    probabilities.append(ZERO_REPLACE)
            else:
                batch_node_id.append(node._id)
                batch_labeled_node_id.append(labeled_node_id)
                batch_lengths.append(shared)

        calc_prob = length_classifier.get_batch_probabilities(batch_lengths,
                                                              batch_node_id,
                                                              batch_labeled_node_id)
        inf_or_nan = np.logical_or(np.isnan(calc_prob), np.isinf(calc_prob))
        calc_prob[inf_or_nan] = INF_REPLACE
        calc_prob[calc_prob == 0.0] = ZERO_REPLACE
            
        return np.append(probabilities, calc_prob)

        
    def identify(self, genome, actual_node):
        node_probabilities = dict() # Probability that a node is a match
        shared_genome_cache = dict()
        for member in self._population.members:
            if member.genome is None:
                continue
            probabilities = self._compare_genome_node(member, genome,
                                                      shared_genome_cache)
            node_probabilities[member] = np.sum(np.log(probabilities))
        # rank = sorted(list(node_probabilities.items()),
        #               key = lambda x: x[1])
        # potential_node = rank[-1][0]
        # import pdb
        # pdb.set_trace()
        # potential_node_probs = self._compare_genome_node(potential_node,
        #                                                  genome,
        #                                                  shared_genome_cache)
        # actual_node_probs = self._compare_genome_node(actual_node,
        #                                               genome,
        #                                               shared_genome_cache)

        potential_node = max(node_probabilities.items(),
                             key = lambda x: x[1])[0]
        return get_sibling_group(potential_node)

def get_sibling_group(node):
    """
    Returns the set containing node and all its full siblings
    """
    return set(node.mother.children).intersection(node.father.children)
