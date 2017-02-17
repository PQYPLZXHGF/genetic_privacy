from collections import namedtuple

# import pyximport; pyximport.install()
import numpy as np

from classify_relationship import (LengthClassifier,
                                   shared_segment_length_genomes)
from util import recent_common_ancestor, first_missing_ancestor

ProbabilityData = namedtuple("ProbabilityData", ["start_i", "stop_i",
                                                 "probabilities"])
MINIMUM_LABELED_NODES = 5
INF = float("inf")
INF_REPLACE = 1.0
UNEXPECTED_IBD = 0.03

import pdb

def calc_for_pair(node_a, node_b, length_classifier, shared_map, id_map,
                  generation_map, unexpected = UNEXPECTED_IBD):
    nonzero_ibd_count = sum(1 for x in shared_map.values() if x > 0)
    print("Nonzero IBD with {} labeled nodes".format(nonzero_ibd_count))
    for labeled_node_id in length_classifier._labeled_nodes:
        labeled_node = id_map[labeled_node_id]
        shared = shared_map[labeled_node_id]
        p_a = length_classifier.get_probability(shared, node_a._id,
                                                labeled_node_id)
        a_cryptic = False
        b_cryptic = False
        if (node_a._id, labeled_node_id) in length_classifier:
            shape, scale, a_zero_prob =  length_classifier._distributions[node_a._id, labeled_node_id]
            a_mean = shape * scale
        else:
            a_cyptic = True
            a_mean = float("NaN")
            a_zero_prob = float("NaN")
        p_b = length_classifier.get_probability(shared, node_b._id,
                                                labeled_node_id)
        if (node_b._id, labeled_node_id) in length_classifier:
            shape, scale, b_zero_prob =  length_classifier._distributions[node_b._id, labeled_node_id]
            b_mean = shape * scale
        else:
            b_cryptic = True
            b_mean = float("NaN")
            b_zero_prob = float("NaN")
        assert p_a != 0.0 and p_b != 0.0
        if p_a == p_b:
            continue
        print("labeled node id: {}".format(labeled_node_id))
        print("IBD: {:.5e}".format(shared))
        print("p_guessed: {:.5e} p_actual: {:.5e}".format(p_a, p_b))
        print("guessed ecdf: {}. actual ecdf {}".format(a_cryptic,
                                                             b_cryptic))
        rca_a = recent_common_ancestor(node_a, labeled_node, generation_map)
        rca_b = recent_common_ancestor(labeled_node, node_b, generation_map)
        print("guessed rca {} actual rca {}".format(rca_a[1], rca_b[1]))
        # print("mean of guessed: {:.5e} mean of actual: {:.5e}".format(a_mean, b_mean))
        # print("Zero prob of guessed: {:.5e} Zero prob of actual: {:.5e}".format(a_zero_prob, b_zero_prob))
        p_a = None
        p_b = None
        a_mean = None
        b_mean = None
        a_zero_prob = None
        b_zero_prob = None

class BayesDeanonymize:
    def __init__(self, population, classifier = None):
        self._population = population
        if classifier is None:
            self._length_classifier = LengthClassifier(population, 1000)
        else:
            self._length_classifier = classifier
        # self.__remove_erroneous_labeled()

    def __remove_erroneous_labeled(self):
        print("Removing erroneous labeled nodes")
        id_map = self._population.id_mapping
        labeled_nodes = [id_map[labeled_node_id] for labeled_node_id
                         in self._length_classifier._labeled_nodes]
        to_remove = set()
        for labeled_node in labeled_nodes:
            missing = first_missing_ancestor(labeled_node)
            if missing < 1:
                to_remove.add(labeled_node._id)
        new_labeled = set(self._length_classifier._labeled_nodes) - to_remove
        print("Removeing {} labeled nodes."
              " New size is {}.".format(len(to_remove), len(new_labeled)))
        self._length_classifier._labeled_nodes = list(new_labeled)
                

    def identify(self, genome, actual_node, population,
                 unexpected = UNEXPECTED_IBD):
        node_probabilities = dict() # Probability that a node is a match
        shared_map = dict()
        id_map = self._population.id_mapping
        length_classifier = self._length_classifier
        for labeled_node_id in length_classifier._labeled_nodes:
            labeled_node = id_map[labeled_node_id]
            s = shared_segment_length_genomes(genome, labeled_node.genome,
                                              5000000)
            shared_map[labeled_node_id] = s

        node_data = dict()
        batch_node_id = []
        batch_labeled_node_id = []
        batch_lengths = []
        cryptic_indices = dict()
        batch_cryptic_lengths = []
        batch_cryptic_labeled_id = []
        distributions = length_classifier._distributions
        nodes = (member for member in self._population.members
                 if member.genome is not None)
        for node in nodes:
            node_probs = []
            node_start_i = len(batch_node_id)
            node_id = node._id
            cryptic_start_i = len(batch_cryptic_lengths)
            for labeled_node_id, shared in shared_map.items():
                # labeled_node = id_map[labeled_node_id]
                # shared = shared_map[labeled_node]
                # shared = shared_map[labeled_node_id]
                if (node_id, labeled_node_id) not in distributions:
                    batch_cryptic_lengths.append(shared)
                    batch_cryptic_labeled_id.append(labeled_node_id)
                else:                    
                    batch_node_id.append(node_id)
                    batch_labeled_node_id.append(labeled_node_id)
                    batch_lengths.append(shared)
            cryptic_stop_i = len(batch_cryptic_lengths)
            node_stop_i = len(batch_node_id)
            node_data[node] = ProbabilityData(node_start_i, node_stop_i,
                                              node_probs)
            cryptic_indices[node] = (cryptic_start_i, cryptic_stop_i)

        calc_prob = length_classifier.get_batch_probability(batch_lengths,
                                                            batch_node_id,
                                                            batch_labeled_node_id)
        # cryptic_prob = length_classifier.get_batch_cryptic_ecdf(batch_cryptic_lengths)
        cryptic_prob = length_classifier.get_batch_smoothing(batch_cryptic_lengths)
        # cryptic_prob = length_classifier.get_batch_ecdf(batch_cryptic_lengths,
        #                                                 batch_cryptic_labeled_id)
        node_probabilities = dict()
        for node, prob_data in node_data.items():
            cryptic_start_i, cryptic_stop_i = cryptic_indices[node]
            if node == actual_node:
                pass
                # import pdb
                # pdb.set_trace()
            node_calc = calc_prob[prob_data.start_i:prob_data.stop_i]
            node_cryptic = cryptic_prob[cryptic_start_i:cryptic_stop_i]
            # log_prob = (np.sum(np.log(node_calc)) +
            #             np.sum(np.log(prob_data.probabilities)))
            log_prob = (np.sum(np.log(node_calc)) +
                        np.sum(np.log(node_cryptic)))
            node_probabilities[node] = log_prob
        potential_node = max(node_probabilities.items(),
                             key = lambda x: x[1])[0]
        # common_ancestor = recent_common_ancestor(potential_node, actual_node,
        #                                          population.node_to_generation)
        # print("Actual node and guessed node have a common ancestor {} generations back.".format(common_ancestor[1]))
        # calc_for_pair(potential_node, actual_node, length_classifier, shared_map, id_map, population.node_to_generation)
        # print("Log probability for guessed {}, log probability for actual {}".format(node_probabilities[potential_node], node_probabilities[actual_node]))
        # from random import choice
        # random_node = choice(list(member for member in self._population.members
        #                           if member.genome is not None))
        # calc_for_pair(random_node, actual_node, length_classifier, shared_map, id_map)
        # calc_for_pair(random_node, potential_node, length_classifier, shared_map, id_map)
        siblings = get_sibling_group(potential_node)
        return siblings

def get_sibling_group(node):
    """
    Returns the set containing node and all its full siblings
    """
    return set(node.mother.children).intersection(node.father.children)
