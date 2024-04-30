.. ASF documentation master file, created by
   sphinx-quickstart on Tue May 19 16:27:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Active Space Finder Software
============================

The Active Space Finder (ASF) is a set of functions for the (semi-)automatic selection of active
spaces in molecules to be employed with complete active space methods such as CASSCF.
Choosing an appropriate set of active orbitals can be a complicated task, which
requires a significant amount of expertise. We have set out to develop a tool to make such
calculations easier and more accessible to both expert and non-expert users.

Employing early-stage quantum devices in the so-called Noisy Intermediate-Scale Quantum (NISQ)
era for applications in quantum chemistry requires the majority of problems to be reduced to their
most important degrees of freedom. The ASF sets out to determine those parts of a molecular system
that need to be treated at the most advanced quantum mechanical level.

The source code is available from: https://github.com/HQSquantumsimulations/ActiveSpaceFinder .
The ASF package is developed at
`HQS Quantum Simulations <https://quantumsimulations.de/>`_, and it is released under the
`Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

ASF is developed with the principles of automation, flexibility and ease of development in mind.
Coding standards are ensured through the application of `black <https://black.readthedocs.io/en/stable/>`_, `ruff <https://docs.astral.sh/ruff/>`_, and `mypy <https://mypy.readthedocs.io/en/stable/>`_.

The easiest way to use the ASF is by calling one of a few wrapper functions, which executes a generic procedure
to determine the active space with minimal requirements for user intervention. More advanced users
can design their own selection procedure with the individual components provided by the ASF
package.

Features
--------

- Active space selection using fast, approximate DMRG calculations.
- The active space selection procedure identifies correlation partner orbitals.
- Simple and easy-to-use wrapper functions for ground and excited states.
- Advanced features to suggest multiple active spaces for a single geometry.
- Perturbative screening methods to identify an orbital subset for the DMRG calculation.
- Visualization functionality.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Home <self>
   installation
   background
   examples
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
