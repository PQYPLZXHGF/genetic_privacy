Getting Started
===============

Requirement
-----------

* Python 3.4+ or Python 3.3+ with [enum](https://pypi.python.org/pypi/enum34) package
* scipy/numpy
* Cython



Setup
-----

* run `fetch.sh` in `data/recombination_rates` directory. This will fetch the appropriate recombination data from the HapMap project.
* run `python3 setup.py build_ext --inplace` in the predict directory. This will need to be run whenever any of the .pyx files are modified.

Running
=======

Note: For the most part, executables are in the `predict` directory.

To generate a population use the `generate_population.py` script. For example if you cd into the `predict` directory and run:
`python3 generate_population.py ../data/tree_file ../data/recombination_rates/ --generation_size 1000 --num_generations 10 --output population.pickle`
A population with 10 generations each with 1000 members will be generated and saved to population.pickl with Python's pickle format.
