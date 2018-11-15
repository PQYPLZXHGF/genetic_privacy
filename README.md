Getting Started
===============

Requirement
-----------

* Python 3.4+ or Python 3.3+ with [enum](https://pypi.python.org/pypi/enum34) package
* scipy/numpy
* Cython
* Statsmodels
* progressbar2



Setup
-----

* run `fetch.sh` in `data/recombination_rates` directory. This will
  fetch the appropriate recombination data from the HapMap project.
* run `python3 setup.py build_ext --inplace` in the predict
  directory. This will need to be run whenever any of the .pyx files
  are modified.

Running
=======

Notes:
* For the most part, executables are in the `predict` directory.
* Some of the script names don't make sense, because the files were
  named at a time when many functions were group together, then split
  out into other files.

Work flow
---------

The typical work flow is a three step process

1. Generate population - `python3 generate_population.py --help`
2. Gather samples - `python3 run_classify_relationship.py --help`
3. Identify - `python3 evaluate_deanonymize.py --help`

Commands
--------

### Generate population

To generate a population use the `generate_population.py` script. For
example if you cd into the `predict` directory and run: `python3
generate_population.py ../data/tree_file ../data/recombination_rates/
--generation_size 1000 --num_generations 10 --output
population.pickle` A population with 10 generations each with 1000
members will be generated and saved to population.pickle with Python's
pickle format.


### Gather samples

To run experiments to collect data, run the command: `python3
run_classify_relationship.py population.pickle work_dir 100
--num_labeled_nodes 150 --output_pickle distributions.pickle`


This command will pick 150 nodes from the last generation and mark
them as "labeled". All other nodes in the last 3 generations are
considered "unlabeled". Then it will perform 100 experiments to sample
from the simulated empirical distributions from the (labeled,
unlabeled) pairs. After collecting 100 samples, it will fit a gamma
distribution for each of these pairs to the empirical distribution.The
total number of data points will be at most 100 samples multiplied by
the number of (labeled, unlabeled) pairs.

Some pruning is done. Values are not sampled for pairs that are
related more than `--gen_back` generations.

This command will delete the `work_dir` directory if it already
exists, and create it if it doesn't (Unless the `--recover` option is
used).

This command tends to take a long time to run. If this process is
interrupted it can be resumed using `--recover`. If you provide the
recover option, the `--num_labeled_nodes` option will be ignored, as
the labeled nodes will be determined by `work_dir`. Recovering will
try to do `num_iteration` new iterations, on top of what may already
be in the `work_dir`. If `num_iterations` is 0, no experiments will be
run, but rather the distributions will be calculated immediately.


### Identify

The final step is identifying unlabeled individuals.

Running `python3 evaluate_deanonymize.py population.pickle
distributions.pickle -n 10` will try to identify 10 random unlabeled
individuals in the population.
