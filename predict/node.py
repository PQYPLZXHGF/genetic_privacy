from random import choice

from sex import Sex, SEXES

class NodeGenerator:
    """
    We use a node generator so that nodes point to each other by id
    rather than by reference. Pickle does not handle highly recursive
    datastructures well. This is a performance optimization that
    should allow pickle and other modules that use recursion to work
    on nodes and populations without overflowing the stack.  The IDs
    are also usefull when using multiprocessing, as just the Node's id
    can be passed from one process to the other.
    """
    def __init__(self):
        self._id = 0
        self._mapping = dict()
        
    def generate_node(self, father = None, mother = None,
                      suspected_father = None, suspected_mother = None,
                      sex = None):
        node_id = self._id
        self._id += 1
        if father is not None:
            father_id = father._id
        else:
            father_id = None
        if suspected_father is not None:
            suspected_father_id = suspected_father._id
        elif father is not None:
            suspected_father_id = father._id
        else:
            suspected_father_id = None
            
        if mother is not None:
            mother_id = mother._id
        else:
            mother_id = None
        if suspected_mother is not None:
            suspected_mother_id = suspected_mother._id
        elif mother is not None:
            suspected_mother_id = mother._id
        else:
            suspected_mother_id = None

        node = Node(self, node_id, father_id, mother_id, suspected_father_id,
                    suspected_mother_id, sex = sex)
        self._mapping[node_id] = node
        return node
    
    @property
    def mapping(self):
        return self._mapping

class Node:
    """
    The node class represents a person in our genealogy.
    The mother and father properties are the true biological mother and father

    The suspected_mother and suspected_father are the individuals the
    'attacker' thinks are the mother and
    father. suspected_mother/father may be the true biological mother
    and father, or they may be other nodes in the geneology. They can
    be different due to errors in the records the attacker has, which
    can be caused by non-paternity events or adoption for example.
    """
    def __init__(self, node_generator, self_id, father_id = None,
                 mother_id = None, suspected_father_id = None,
                 suspected_mother_id = None, sex = None):
        self.genome = None
        self._node_generator = node_generator
        self._mother_id = mother_id
        if suspected_mother_id is None:
            self._suspected_mother_id = mother_id
        else:
            self._suspected_mother_id = suspected_mother_id
        if suspected_father_id is None:
            self._suspected_father_id = father_id
        else:
            self._suspected_father_id = suspected_father_id
        self._father_id = father_id
        self._id = self_id
        if isinstance(sex, Sex):
            self.sex = sex
        else:
            self.sex = choice(SEXES)
        self._children = []
        self._suspected_children = []
        self._resolve_parents()
        if self.mother is not None:
            assert self.mother.sex == Sex.Female
            self.mother._children.append(self._id)
        if self.suspected_mother is not None:
            assert self.suspected_mother.sex == Sex.Female
            self.mother._suspected_children.append(self._id)
        if self.father is not None:
            assert self.father.sex == Sex.Male
            self.father._children.append(self._id)
        if self.suspected_father is not None:
            assert self.suspected_father.sex == Sex.Male
            self.father._suspected_children.append(self._id)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["mother"]
        del state["father"]
        del state["suspected_mother"]
        del state["suspected_father"]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Parents are resolved by PopulationUnpickler after all Node
        # objects are created

    def _resolve_parents(self):
        """
        This method resolves the respective parent ids in to objects.
        This is usually done in the constructor or after unpickling a node
        object.
        """
        mapping = self._node_generator._mapping
        if self._mother_id is not None:
            self.mother = mapping[self._mother_id]
        else:
            self.mother = None
        if self._father_id is not None:
            self.father = mapping[self._father_id]
        else:
            self.father = None
            
        if self._suspected_father_id is not None:
            self.suspected_father = mapping[self._suspected_father_id]
        else:
            self.suspected_father = self.father
        if self._suspected_mother_id is not None:
            self.suspected_mother = mapping[self._suspected_mother_id]
        else:
            self.suspected_mother = self.mother
            
    @property
    def mapping(self):
        """
        Returns a the dictionary mapping node id -> node object
        """
        return self._node_generator._mapping

    @property
    def children(self):
        """
        The true children of this node
        """
        return [self._node_generator._mapping[node_id]
                for node_id in self._children]

    @property
    def suspected_children(self):
        """
        The suspected children of this node
        """
        return [self._node_generator._mapping[node_id]
                for node_id in self._suspected_children]

    @property
    def node_generator(self):
        return self._node_generator
