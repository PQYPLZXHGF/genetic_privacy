from random import choice
from collections import defaultdict

import matplotlib.pyplot as plt
# from matplotlib import rc, rcParams
import numpy as np
from scipy.interpolate import UnivariateSpline

from scipy.stats import beta, gamma, poisson, exponweib
# LENGTHS_FILENAME = "/media/paul/Storage/scratch/lengths/70004"
# LENGTHS_FILENAME = "/media/paul/Fast Storage/work_dirs/population_10000_monog_80_2/94863"
# LENGTHS_FILENAME = "/media/paul/Fast Storage/work_dirs/population_10000_monog_80_non_paternity_005_perturb/84100"
LENGTHS_FILENAME = "/media/paul/Fast Storage/work_dirs/population_10000_monog_80_non_paternity_005/84100"



# rcParams.update({'font.size': 24})

def get_lengths(lengths_filename):
    lengths = defaultdict(list)
    with open(lengths_filename, "r") as lengths_file:
        for line in lengths_file:
            try:
                node_id, length = line.split("\t")
            except ValueError:
                continue
            node_id = int(node_id)
            length = int(length)
            if node_id > 100000 or length > 10000000000:
                continue
            lengths[node_id].append(length)
    return lengths


def multi_probs(lengths_list, labels = None):
    for i, lengths in enumerate(lengths_list):
        lengths.sort()
        prob, x = np.histogram(lengths, bins = 15)
        x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
        f = UnivariateSpline(x, prob)
        if labels is not None:
            plt.plot(x, f(x), label = labels[i])
        else:
            plt.plot(x, f(x))
    if labels is not None:
        plt.legend(loc = "upper right")
    plt.ylabel("Simliarity function probability", size = 20)
    plt.xlabel("Similarity function value", size = 20)
    plt.show()

def multi_probs_2(lengths_list, labels = None):
    for i, lengths in enumerate(lengths_list):
        lengths.sort()
        weights = np.ones_like(lengths)/float(len(lengths))
        if labels is not None:
            plt.hist(lengths, bins = 20, weights = weights,
                     alpha = 0.5, normed = False,
                     label = labels[i])
        else:
            plt.hist(lengths, bins = 20, weights = weights,
                     alpha = 0.5, normed = False)
    if labels is not None:
        plt.legend(loc = "upper right")
    plt.ylabel("Simliarity function probability", size = 20)
    plt.xlabel("Similarity function value", size = 20)
    plt.show()
            
    
def plot_histogram(lengths):
    lengths.sort()
    # weights = np.ones_like(lengths)/float(len(lengths))
    # plt.hist(lengths, weights=weights, bins = 20)
    prob, x = np.histogram(lengths, bins = 15)
    x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
    f = UnivariateSpline(x, prob)
    plt.plot(x, f(x))
    
    plt.ylabel("Simliarity function probability", size = 20)
    plt.xlabel("Similarity function value", size = 20)
    # plt.hist(lengths, normed = True, bins = 20)

    # np_lengths = np.array(lengths)
    # beta_pdf = beta(*beta.fit(np_lengths)).pdf(np_lengths)
    # gamma_pdf = gamma(*gamma.fit(np_lengths)).pdf(np_lengths)

    # beta_plot = plt.plot(lengths, beta_pdf, label = "Beta")
    # gamma_plot = plt.plot(lengths, gamma_pdf, label = "Gamma")
    # plt.legend(loc = "upper right")
    plt.show()

def plot_histogram_drop_zero(lengths, name = "default"):
    lengths.sort()
    lengths = np.array(lengths, dtype = np.uint32)
    lengths = lengths[lengths != 0]
    weights = np.ones_like(lengths)/float(len(lengths))
    plt.hist(lengths, weights=weights, bins = 20)
    # prob, x = np.histogram(lengths, bins = 15)
    # x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
    # f = UnivariateSpline(x, prob)
    # plt.plot(x, f(x))
    
    # plt.ylabel("Simliarity function probability", size = 20)
    # plt.xlabel("Similarity function value", size = 20)
    # plt.hist(lengths, normed = True, bins = 20)

    # np_lengths = np.array(lengths)
    # beta_pdf = beta(*beta.fit(np_lengths)).pdf(np_lengths)
    # gamma_pdf = gamma(*gamma.fit(np_lengths)).pdf(np_lengths)

    # beta_plot = plt.plot(lengths, beta_pdf, label = "Beta")
    # gamma_plot = plt.plot(lengths, gamma_pdf, label = "Gamma")
    # plt.legend(loc = "upper right")
    # plt.show()
    plt.savefig("perturb_vs_nothing/84100/nothing/{}.svg".format(name))
    plt.close()

lengths_vectors = get_lengths(LENGTHS_FILENAME)
sorted_nodes = sorted(list(lengths_vectors.keys()))
for node in sorted_nodes:
    lengths = lengths_vectors[node]
    print(node)
    plot_histogram_drop_zero(lengths, node)
    # plot_histogram(lengths)
# plot_histogram(lengths_vectors[77850])
# multi_probs_2([lengths_vectors[77850], lengths_vectors[70003]],
#             ["first cousin", "full sibling"])
# for node_id, lengths in lengths_vectors.items():
#     plot_histogram(lengths)
    
