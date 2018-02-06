class ExpansionData:
    def __init__(self, start_labeled):
        self.start_labeled = start_labeled
        self.added = []

    def add_round(self, to_add):
        self.added.extend(to_add)

    def adjust_genomes(self, population):
        id_mapping = population.id_mapping
        for result in self.added:
            if result.correct:
                continue
            identified = id_mapping[result.identified_node]
            target = id_mapping[result.target_node]
            identified.genome = target.genome

    @property
    def labeled_nodes(self):
        ret = list(self.start_labeled)
        ret.extend(result.identified_node for result in self.added)
        return ret
