from functools import reduce
from operator import mul
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
        length_classifier = self._length_classifier
        for labeled_node in length_classifier._labeled_nodes:
            if labeled_node in cache:
                shared = cache[labeled_node]
            else:
                shared = shared_segment_length_genomes(genome,
                                                       labeled_node.genome,
                                                       0)
                cache[labeled_node] = shared
            if (node, labeled_node) not in length_classifier:
                if shared == 0:
                    prob = INF_REPLACE
                else:
                    prob = ZERO_REPLACE
            else:
                prob = length_classifier.get_probability(shared, node,
                                                         labeled_node)
                if prob > 1 or isnan(prob):
                    prob = INF_REPLACE
                if prob == 0:
                    prob = ZERO_REPLACE
                
            probabilities.append(prob)
            
        return np.array(probabilities)

        
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
