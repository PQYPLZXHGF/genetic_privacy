from math import isnan
from collections import namedtuple

# import pyximport; pyximport.install()
import numpy as np

from classify_relationship import (LengthClassifier,
                                   shared_segment_length_genomes,
                                   ZERO_REPLACE)

ProbabilityData = namedtuple("ProbabilityData", ["start_i", "stop_i",
                                                 "probabilities"])
MINIMUM_LABELED_NODES = 5
INF = float("inf")
INF_REPLACE = 1.0

class BayesDeanonymize:
    def __init__(self, population, classifier = None):
        self._population = population
        if classifier is None:
            self._length_classifier = LengthClassifier(population, 1000)
        else:
            self._length_classifier = classifier

        
    def identify(self, genome, actual_node):
        node_probabilities = dict() # Probability that a node is a match
        shared_map = dict()
        id_map = self._population.id_mapping
        length_classifier = self._length_classifier
        for labeled_node_id in length_classifier._labeled_nodes:
            labeled_node = id_map[labeled_node_id]
            s = shared_segment_length_genomes(genome, labeled_node.genome, 0)
            shared_map[labeled_node] = s

        node_data = dict()
        batch_node_id = []
        batch_labeled_node_id = []
        batch_lengths = []
        nodes = (member for member in self._population.members
                 if member.genome is not None)
        for node in nodes:
            node_probs = []
            node_start_i = len(batch_node_id)
            for labeled_node_id in length_classifier._labeled_nodes:
                labeled_node = id_map[labeled_node_id]
                shared = shared_map[labeled_node]
                if (node._id, labeled_node_id) not in length_classifier:
                    if shared == 0:
                        node_probs.append(INF_REPLACE)
                    else:
                        node_probs.append(ZERO_REPLACE)
                else:
                    batch_node_id.append(node._id)
                    batch_labeled_node_id.append(labeled_node_id)
                    batch_lengths.append(shared)
            node_stop_i = len(batch_node_id)
            node_data[node] = ProbabilityData(node_start_i, node_stop_i,
                                              node_probs)
        calc_prob = length_classifier.get_batch_probability(batch_lengths,
                                                            batch_node_id,
                                                            batch_labeled_node_id)
        inf_or_nan = np.logical_or(np.isnan(calc_prob), np.isinf(calc_prob))
        calc_prob[inf_or_nan] = INF_REPLACE
        calc_prob[calc_prob == 0.0] = ZERO_REPLACE
        node_probabilities = dict()
        for node, prob_data in node_data.items():
            if node == actual_node:
                import pdb
                pdb.set_trace()
            node_calc = calc_prob[prob_data.start_i:prob_data.stop_i]
            log_prob = (np.sum(np.log(node_calc)) +
                        np.sum(np.log(prob_data.probabilities)))
            node_probabilities[node] = log_prob
        potential_node = max(node_probabilities.items(),
                             key = lambda x: x[1])[0]
        return get_sibling_group(potential_node)

def get_sibling_group(node):
    """
    Returns the set containing node and all its full siblings
    """
    return set(node.mother.children).intersection(node.father.children)
