from collections import namedtuple
from heapq import nlargest

# import pyximport; pyximport.install()
import numpy as np

from classify_relationship import (LengthClassifier,
                                   shared_segment_length_genomes)
from data_logging import write_log
from util import first_missing_ancestor, all_related

ProbabilityData = namedtuple("ProbabilityData", ["start_i", "stop_i",
                                                 "cryptic_start_i",
                                                 "cryptic_stop_i"])

RawIdentified = namedtuple("RawIdentified", ["sibling_group",
                                             "ln_ratio",
                                             "identified_node"])
MINIMUM_LABELED_NODES = 5
INF = float("inf")
INF_REPLACE = 1.0

class BayesDeanonymize:
    def __init__(self, population, classifier = None, only_related = False,
                 search_generations_back = 7):
        self._population = population
        if classifier is None:
            self._length_classifier = LengthClassifier(population, 1000)
        else:
            self._length_classifier = classifier
        self._only_related = only_related
        if only_related:
            self._search_generations_back = search_generations_back
            self._compute_related()
        self._restrict_search_nodes = None
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

    def _to_search(self, shared_list):
        labeled = set(self._length_classifier._labeled_nodes)
        genome_nodes = (member for member in self._population.members
                        if (member.genome is not None and
                            member._id not in labeled))
        if not self._only_related:
            return genome_nodes
        id_map = self._population.id_mapping
        genome_nodes = set(genome_nodes)
        potential_nodes = set()
        for labeled_node_id, shared in shared_list:
            if shared > 0:
                labeled_node = id_map[labeled_node_id]
                related = self._labeled_related[labeled_node]
                potential_nodes.update(related)
        if self._restrict_search_nodes is not None:
            return potential_nodes.intersection(self._restrict_search_nodes)
        else:
            return potential_nodes

    def _add_node_id_relatives(self, node_id, nodes):
        id_map = self._population.id_mapping
        labeled_node = id_map[node_id]
        related = all_related(labeled_node, True, self._search_generations_back)
        self._labeled_related[labeled_node] = related.intersection(nodes)

    def _compute_related(self):
        nodes = set(member for member in self._population.members
                    if member.genome is not None)
        self._labeled_related = dict()
        length_classifier = self._length_classifier
        for labeled_node_id in length_classifier._labeled_nodes:
            self._add_node_id_relatives(labeled_node_id, nodes)

    def add_labeled_node_id(self, node_id):
        self._length_classifier._labeled_nodes.append(node_id)
        if self._only_related:
            nodes = list(member for member in self._population.members
                         if member.genome is not None)
            self._add_node_id_relatives(node_id, nodes)

    def restrict_search(self, nodes):
        self._restrict_search_nodes = set(nodes)

    def identify(self, genome, actual_node, ibd_threshold = 5000000):
        node_probabilities = dict() # Probability that a node is a match
        id_map = self._population.id_mapping
        length_classifier = self._length_classifier
        shared_list = []
        for labeled_node_id in length_classifier._labeled_nodes:
            labeled_node = id_map[labeled_node_id]
            s = shared_segment_length_genomes(genome,
                                              labeled_node.suspected_genome,
                                              ibd_threshold)
            shared_list.append((labeled_node_id, s))

        node_data = dict()
        batch_node_id = []
        batch_labeled_node_id = []
        batch_lengths = []
        batch_cryptic_lengths = []
        # This is done for performance reasons, as appending to this
        # list is the hottest part of the loop.
        append_cryptic = batch_cryptic_lengths.append
        distributions = length_classifier._distributions
        # Set membership testing is faster than dictionary key
        # membership testing, so we use a set.
        distribution_members = set(distributions.keys())
        nodes = self._to_search(shared_list)
        for node in nodes:
            node_start_i = len(batch_node_id)
            node_id = node._id
            cryptic_start_i = len(batch_cryptic_lengths)
            for labeled_node_id, shared in shared_list:
                if (node_id, labeled_node_id) not in distribution_members:
                    append_cryptic(shared)
                else:                    
                    batch_node_id.append(node_id)
                    batch_labeled_node_id.append(labeled_node_id)
                    batch_lengths.append(shared)
            cryptic_stop_i = len(batch_cryptic_lengths)
            node_stop_i = len(batch_node_id)
            node_data[node] = ProbabilityData(node_start_i, node_stop_i,
                                              cryptic_start_i, cryptic_stop_i)

        if len(batch_lengths) > 0:
            calc_prob = length_classifier.get_batch_probability(batch_lengths,
                                                                batch_node_id,
                                                                batch_labeled_node_id)
        else:
            calc_prob = []
        cryptic_prob = length_classifier.get_batch_smoothing(batch_cryptic_lengths)

        # index_data = {node._id: tuple(indices)
        #               for node, indices in node_data.items()}
        # siblings = {node._id for node in get_sibling_group(actual_node)}
        # to_dump = {"actual_node_id": actual_node._id,
        #            "calc_prob": calc_prob,
        #            "cryptic_lengths": batch_cryptic_lengths,
        #            "siblings": siblings,
        #            "index_data": index_data}
        # output_filename = "/media/paul/Fast Storage/optimize_data/{}.pickle".format(actual_node._id)
        # with open(output_filename, "wb") as pickle_file:
        #     dump(to_dump, pickle_file)
        node_probabilities = dict()
        for node, prob_data in node_data.items():
            start_i, stop_i, cryptic_start_i, cryptic_stop_i = prob_data
            if node == actual_node:
                pass
                # import pdb
                # pdb.set_trace()
            node_calc = calc_prob[start_i:stop_i]
            node_cryptic = cryptic_prob[cryptic_start_i:cryptic_stop_i]
            log_prob = (np.sum(np.log(node_calc)) +
                        np.sum(np.log(node_cryptic)))
            node_probabilities[node] = log_prob
        # potential_node = max(node_probabilities.items(),
        #                      key = lambda x: x[1])[0]
        write_log("identify", {"node": actual_node._id,
                               "probs": {node._id: prob
                                         for node, prob
                                         in node_probabilities.items()}})
        potential_nodes = nlargest(8, node_probabilities.items(),
                                   key = lambda x: x[1])
        top, top_log_prob = potential_nodes[0]
        sibling_group = get_sibling_group(top)
        for node, log_prob in potential_nodes[1:]:
            if node in sibling_group:
                continue
            next_node = node
            next_log_prob = log_prob
            break
        else:
            next_node, next_log_prob = potential_nodes[1]
                
        log_ratio  = top_log_prob - next_log_prob
        # log_data = {"actual_node_id": actual_node._id,
        #             "prob_indices": prob_data,
        #             "calc_prob": calc_prob,
        #             "cryptic_prob": cryptic_prob
        #             "sibling_group": [node._id for node in sibling_group]}
        # write_log("run_data", log_data)
        return RawIdentified(sibling_group, log_ratio, top)
        # return (sibling_group, log_ratio)
        # return set(chain.from_iterable(get_sibling_group(potential[0])
        #                                for potential in potential_nodes))

def get_sibling_group(node):
    """
    Returns the set containing node and all its full siblings
    """
    if node.mother is None or node.father is None:
        return set([node])
    return set(node.mother.children).intersection(node.father.children)
