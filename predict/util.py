from itertools import combinations, product, chain
from random import sample
from collections import deque

def all_ancestors(node):
    ancestors = set()
    current_generation = set([node])
    while len(current_generation) > 0:
        mothers = set(node.mother for node in current_generation
                      if node.mother is not None)
        fathers = set(node.father for node in current_generation
                      if node.father is not None)
        current_generation = set(chain(mothers, fathers))
        ancestors.update(current_generation)
    return ancestors

def recent_common_ancestor(node_a, node_b, generation_map):
    if node_a == node_b:
        return (node_a, 0)
    if node_a.mother == node_b.mother:
        return (node_a.mother, 1)
    if node_a.father == node_b.father:
        return (node_a.father, 1)
    a_ancestors = all_ancestors(node_a)
    b_ancestors = all_ancestors(node_b)
    common_ancestors = a_ancestors.intersection(b_ancestors)
    if len(common_ancestors) == 0:
        return (None, None)
    recent_generation = -1
    ancestor = None
    for common_ancestor in common_ancestors:
        ancestor_generation = generation_map[common_ancestor]
        if ancestor_generation > recent_generation:
            recent_generation = ancestor_generation
            ancestor = common_ancestor
    younger_pair = max(generation_map[node_a], generation_map[node_b])    
    return (ancestor, abs(younger_pair - recent_generation))

def get_sample_of_cousins(population, distance, percent_ancestors = 0.1,
                          percent_descendants = 0.1):
    """
    return a sample of pairs of individuals whos most recent common
    ancestor is exactly generations back.
    """
    assert 0 < distance < len(population.generations)
    assert 0 < percent_descendants <= 1
    assert 0 < percent_ancestors <= 1
    common_ancestors = population.generations[-(distance + 1)].members
    last_generation = set(population.generations[-1].members)
    ancestors_sample = sample(common_ancestors,
                              int(len(common_ancestors) * percent_ancestors))
    pairs = []
    for ancestor in ancestors_sample:
        temp_pairs = descendants_with_common_ancestor(ancestor, last_generation)
        temp_pairs = list(temp_pairs)
        pairs.extend(sample(temp_pairs,
                            int(len(temp_pairs) * percent_descendants)))
    return pairs

def descendants_of(node):
    descendants = set()
    to_visit = list(node.children)
    while len(to_visit) > 0:
        ancestor = to_visit.pop()
        descendants.add(ancestor)
        to_visit.extend(ancestor.children)
    return descendants

def descendants_with_common_ancestor(ancestor, generation_members):
    """
    Returns pairs of individuals descendent from ancestor in the given
    generation who have ancestor as their most recent ancestor.
    """
    # Find the descendents of the children, remove the pairwise
    # intersection, and return pairs from different sets.
    ancestor_children = ancestor.children
    if len(ancestor_children) < 2:
        return []
    if generation_members.issuperset(ancestor_children):
        # Depth is only 1 generation, so return all combinations of children.
        return combinations(ancestor_children, 2)
    descendant_sets = [descendants_of(child).intersection(generation_members)
                       for child in ancestor_children]    
    pair_iterables = []
    for descendants_a , descendants_b in combinations(descendant_sets, 2):
        intersection = descendants_a.intersection(descendants_b)
        if len(intersection) > 0:
            # Remove individuals who have a more recent common ancestor
            descendants_a = descendants_a - intersection
            descendants_b = descendants_b - intersection
        pair_iterables.append(product(descendants_a, descendants_b))
    return chain.from_iterable(pair_iterables)

