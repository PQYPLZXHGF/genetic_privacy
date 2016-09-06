from argparse import ArgumentParser
from collections import Counter
from operator import itemgetter

from heapq import nsmallest

parser = ArgumentParser(description="Estimate the number of rounds run.")
parser.add_argument("filename")
args = parser.parse_args()

counter = Counter()
with open(args.filename, "r") as output_file:
    for line in output_file:
        node, length = line.split()
        counter[node] += 1
common_node, common_count = list(counter.most_common(1))[0]
print("Node {} seen {} times".format(common_node, common_count))


uncommon_node, uncommon_count = nsmallest(1, counter.items(),
                                          key=itemgetter(1))[0]
print("Node {} seen {} times".format(uncommon_node, uncommon_count))
