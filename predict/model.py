from enum import Enum
from math import floor
from random import shuffle, choice

# From table F1 https://www.census.gov/hhes/families/data/cps2012F.html
# These numbers are not 100% accurate because each count is the count of
# children under 18.
CHILDREN_PROBABILITIES = {0: 45517/80506, 1: 15033/80506, 2: 12999/80506,
                          3: 4967/80506, 4:1990/80506}

class Sex(Enum):
    Male = 0
    Female = 1

SEXES = list(Sex)

class Node:
    def __init__(self, dad = None, mom = None, sex = None):
        self.mom = mom
        self.dad = dad
        if isinstance(sex, Sex):
            self.sex = sex
        else:
            self.sex = choice(SEXES)
        self.children = set()
        if mom is not None:
            mom.children.add(self)
        if dad is not None:
            dad.children.add(self)
        

class Generation:
    def __init__(self, members = None):
        self.members = set(members)

    @property
    def men(self):
        return (person for person in self.members if person.sex == Sex.Male)

    @property
    def women(self):
        return (person for person in self.members if person.sex == Sex.Female)
    
    @property
    def size(self):
        return len(self.members)

def generate_population(size):
    initial_generation = Generation(set(Node() for _ in range(size)))
    return Population(initial_generation)

class Population:
    def __init__(self, initial_generation = None):
        self._generations = []
        if initial_generation is not None:
            self._generations.append(initial_generation)

    def new_generation(self, size = None):
        """
        Generates a new generation of individuals from the previous
        generation.  If size is not passed, the new generation will be
        the same size as the previous generation.
        """
        if size is None:
            size = self._generations[-1].size
        previous_generation = self._generations[-1]
        new_nodes = []
        men = list(previous_generation.men)
        shuffle(men)
        women = list(previous_generation.women)
        pairs = []
        for man in men:
            if len(women) is 0:
                break
            # We go backwards through the list of women, so we pop
            # them off the end of the list.
            for i in range(len(women) - 1, -1, -1):
                if man.mom is None or women[i].mom != man.mom:
                    pairs.append((man, women.pop(i)))
                    break
        min_children = floor(size / len(pairs))
        # Number of families with 1 more than the min number of
        # children. Because only having 2 children per pair only works
        # if there is an exact 1:1 ratio of men to women.
        extra_child = size - min_children * len(pairs)
        for i, (man, woman) in enumerate(pairs):
            if i < extra_child:
                extra = 1
            else:
                extra = 0
            for i in range(min_children + extra):
                new_nodes.append(Node(man, woman))

        SIZE_ERROR = "Generation generated is not correct size. Expected {}, got {}."
        assert len(new_nodes) == size, SIZE_ERROR.format(size, len(new_nodes))
        self._generations.append(Generation(new_nodes))
        
        

    