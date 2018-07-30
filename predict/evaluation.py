from collections import Counter, defaultdict, namedtuple
from datetime import datetime
from pickle import dump
from sys import stdout
from math import sqrt

from data_logging import write_log
from bayes_deanonymize import BayesDeanonymize
from shared_segment_detector import SharedSegmentDetector


IdentifyResult = namedtuple("IdentifyResult", ["target_node",
                                               "sibling_group",
                                               "identified_node",
                                               "ln_ratio",
                                               "correct",
                                               "run_number"])
class Evaluation:
    def __init__(self, population, classifier, labeled_nodes = None,
                 ibd_detector = None, search_related = False):
        self._population = population
        self._classifier = classifier
        if labeled_nodes is not None:
            self.labeled_nodes = labeled_nodes
        if search_related:
            print("Calculating related nodes")
            self._bayes = BayesDeanonymize(population, classifier,
                                           True, search_related)
        else:
            self._bayes = BayesDeanonymize(population, classifier, False)
        if ibd_detector is None:
            ibd_detector = SharedSegmentDetector()
        assert isinstance(ibd_detector, SharedSegmentDetector)
        self._ibd_detector = ibd_detector
        self._run_number = 0
        self.reset_metrics()

    @property
    def accuracy(self):
        total = self.correct + self.incorrect
        return self.correct / total

    def add_labeled_node_id(self, node):
        self._bayes.add_labeled_node_id(node)

    @property
    def labeled_nodes(self):
        return self._classifier._labeled_nodes.copy()

    @labeled_nodes.setter
    def labeled_nodes(self, labeled_nodes):
        self._classifier._labeled_nodes = labeled_nodes

    def restrict_search(self, nodes):
        assert len(nodes) > 0
        self._bayes.restrict_search(nodes)

    def print_metrics(self):
        total = self.correct + self.incorrect
        print("{} correct, {} incorrect, {} total.".format(self.correct,
                                                           self.incorrect,
                                                           total))
        stdout.flush()

        write_log("correct", self.correct)
        write_log("incorrect", self.incorrect)
        write_log("total", total)
        percent_accurate = self.accuracy
        std_dev = sqrt(percent_accurate * (1 - percent_accurate) * total) / total
        print("{}±{:0.3} percent accurate.".format(percent_accurate, std_dev))
        for generation, counter in self.generation_error.items():
            gen_correct = counter["correct"]
            gen_incorrect = counter["incorrect"]
            total = gen_correct + gen_incorrect
            format_string = "For generation {}: {} accuracy, {} total."
            print(format_string.format(generation, gen_correct / total, total))

    def _evaluate_node(self, node):
        raw_identified = self._bayes.identify(node.genome, node,
                                              self._ibd_detector)
        sibling_group, ln_ratio, identified_node = raw_identified
        node_generation = self._population.node_to_generation[node]
        if node in sibling_group:
            self.generation_error[node_generation]["correct"] += 1
            self.correct += 1
            print("correct")
        else:
            self.generation_error[node_generation]["incorrect"] += 1
            print("incorrect")
            self.incorrect += 1
            
        write_log("evaluate", {"target node": node._id,
                               "log ratio": ln_ratio,
                               "identified": set(x._id for x in sibling_group),
                               "run_number": self._run_number})
        stdout.flush()
        return IdentifyResult(node,
                              sibling_group,
                              identified_node,
                              ln_ratio,
                              node in sibling_group,
                              self._run_number)

    def reset_metrics(self):
        # Maps generation -> counter with keys "correct" and "incorrect"
        self.generation_error = defaultdict(Counter)
        self.identify_results = []
        self.correct = 0
        self.incorrect = 0

    def run_evaluation(self, unlabeled):
        # generation_map = population.node_to_generation
        # write_log("labeled_nodes", [node._id for node in labeled_nodes])
        # write_log("target_nodes", [node._id for node in unlabeled])
        print("Attempting to identify {} random nodes.".format(len(unlabeled)),
              flush = True)
        write_log("start time", datetime.now())
        for i, node in enumerate(unlabeled):
            print("Iteration: {}, actual node ID: {}".format(i + 1, node._id))
            self.identify_results.append(self._evaluate_node(node))

        write_log("end time", datetime.now())
        self._run_number += 1
        return self.identify_results

    def _calculate_ibd_segments_for_expansion(self, to_evaluate):
        """
        Calculate IBD segments for nodes to search in expansion round,
        so that we can prioritize prospective nodes with the most IBD
        shared with labeled nodes.
        """
        shared_dict = dict()
        detector = self._ibd_detector
        id_map = self._population.id_mapping
        labeled_nodes = [id_map[node_id] for node_id in self.labeled_nodes]
        for node in to_evaluate:
            shared = [detector.shared_segment_length(node.genome,
                                                     labeled.genome)
                      for labeled in labeled_nodes]
            shared_dict[node] = shared
        self._expanson_shared = shared_dict

    def _expansion_add_labeled(self, new_labeled_node, to_evaluate):
        detector = self._ibd_detector
        for node in to_evaluate:
            shared = detector.shared_segment_length(node.genome,
                                                    new_labeled_node.genome)
            self._expanson_shared[node].append(shared)
    
    def _get_expansion_node(self, candidates):
        shared = self._expanson_shared
        shared_segment_count = []
        for node, ibd_lengths in shared.items():
            if node in candidates:
                nonzero_counts = sum(length > 0 for length in ibd_lengths)
                shared_segment_count.append((node, nonzero_counts))
        if len(shared_segment_count) == 0:
            return None
        return max(shared_segment_count, key = lambda x: x[1])[0]

    def run_expansion_round(self, identify_candidates,
                            expansion_data = None,
                            expansion_filename = None):
        print("Running expansion round.")
        to_evaluate = set(identify_candidates)
        self._calculate_ibd_segments_for_expansion(to_evaluate)
        added = []
        correct_add_count = 0
        i = 0
        exclude = set()
        while len(to_evaluate) > 0:
            # TODO: Turn this into an iterator?
            node = self._get_expansion_node(to_evaluate - exclude)
            if node is None:
                break
            exclude.add(node)
            self.run_evaluation([node])
            result = self.identify_results[-1]
            print("Ratio: {}".format(result.ln_ratio))
            if result.ln_ratio > 9:
                print("Adding node.")
                added.append(result)
                identified = result.identified_node
                self._bayes.add_labeled_node_id(identified._id)
                if result.correct:
                    correct_add_count += 1
                else:
                    identified.suspected_genome = result.target_node.genome
                exclude = set()
                to_evaluate.remove(identified)
                self._expansion_add_labeled(identified, to_evaluate)
                del self._expanson_shared[identified]
            if i % 20 == 0:
                self.print_metrics()
                print("Nodes added this round: {}".format(len(added)))
                print("Correct nodes added: {}".format(correct_add_count))
            if expansion_data and expansion_filename and i % 500 == 0:
                remaining = set(node._id for node in to_evaluate)
                expansion_data.extend_round(added, remaining)
                with open(expansion_filename, "wb") as expansion_file:
                    dump(expansion_data, expansion_file)
                write_log("expansion_data_written", {"current_node": node._id,
                                                     "complete": False})
            i += 1
        write_log("expansion_round", {"added": len(added),
                                      "correct_added": correct_add_count,
                                      "accuracy": self.accuracy})
        self.print_metrics()
        print("Added {} nodes this round.".format(len(added)))
        return added
