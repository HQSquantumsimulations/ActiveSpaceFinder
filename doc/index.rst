.. ASF documentation master file, created by
   sphinx-quickstart on Tue May 19 16:27:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for ASF!
=====================================

The Active Space Finder (ASF) is a set of functions for the (semi-)automatic
selection of active spaces in molecules to be employed with methods such as CASSCF.
Choosing an appropriate set of active orbitals can be a complicated task,
which requires a significant amount of expertise. We have set out to develop a tool to make
such calculations easier and more accessible to both expert and non-expert users.

Employing early-stage quantum devices, in the so-called Noisy-Intermediate-Scale-Quantum (NISQ)
era, for applications in quantum chemistry requires the majority of problems to be reduced to their
most important degrees of freedom. The ASF sets out to determine those parts of a molecular system
that need to be treated at the most advanced quantum mechanical level.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   tutorials/examples
   benchmarks/diradicals
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
