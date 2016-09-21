Getting Started
===============

Requirement
-----------

* Python 3.5+ or Python 3.3+ with enum package
* scipy/numpy
* Cython

Note: For the most part, executable code is in the "predict" directory.

Setup
-----

* run `fetch.sh` in data/recombination_rates` directory. This will fetch the appropriate recombination data from the HapMap project.
* run `python3 setup.py build_ext --inplace` in the predict directory. This will need to be run whenever any of the .pyx files are modified.
