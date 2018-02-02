from itertools import chain

class ExpansionData:
    def __init__(self, start_labeled_nodes):
        self.start_labeled_nodes = list(start_labeled_nodes)
        self._rounds = []

    def add_round(self, to_update):
        self._rounds.append(to_update)

    def apply_to_population(self, population):
        for result in chain.from_iterable(self._rounds):
            if result.correct:
                continue
            
