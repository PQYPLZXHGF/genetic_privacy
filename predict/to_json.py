import json

def to_json(population, labeled_nodes, related_nodes):
    all_nodes = list(population.members)
    all_nodes.sort(key = lambda node: node._id)
    json_nodes = list()
    for node in all_nodes:
        json_node = dict()
        json_node["id"] = node._id
        if node.father is not None:
            json_node["father"] = node.father._id
        if node.mother is not None:
            json_node["mother"] = node.mother._id
        if node.suspected_father is not None:
            json_node["suspected_father"] = node.suspected_father._id
        if node.suspected_mother is not None:
            json_node["suspected_mother"] = node.suspected_mother._id
        if node.twin is not None:
            json_node["twin"] = node.twin._id
        json_nodes.append(json_node)
    json_labeled_nodes = [node._id for node in labeled_nodes]
    json_related = [(a._id, b._id) for a, b in related_nodes]
    population_json = {"nodes": json_nodes,
                       "related": json_related,
                       "labeled": json_labeled_nodes}
    return json.dumps(population_json)
    
