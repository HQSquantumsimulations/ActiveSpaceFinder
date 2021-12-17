Overview
========

The Active Space Finder (ASF) is a set of functions for the (semi-)automatic
selection of active spaces in molecules. ASF aims to automatize the selection of active orbitals,
thus facilitating the black-box application of multi-reference methods such as CASSCF. The easiest
way to use the ASF is by calling a wrapper function, which executes a generic procedure to
determine the active space with minimal requirements for user intervention.

More advanced users can design their own selection procedure with the individual components
provided by the ASF software. The code is developed at
`HQS Quantum Simulations <https://quantumsimulations.de/>`_, and it is released under the
`Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

ASF requires the following software packages:

- Python3 with numpy and scipy
- `PySCF v.1.7.6 <https://github.com/pyscf/pyscf/>`_ for general electronic structure calculations
- `BLOCK v.1.5 <https://github.com/sanshar/Block/>`_ for DMRG calculations
- `Jmol v.14.6.4 <http://jmol.sourceforge.net/>`_ for plotting

ASF is developed with the principles of automation, flexibility and ease of development in mind.
Coding standards are ensured through the application of flake8 and mypy.

At its core, the ASF employs a DMRG calculation with low accuracy settings to determine so-called
single-orbital entropies. These entropies quantify the degree of entanglement for each orbital,
and are used to select an active space.

Since the number of orbitals that can be used for a DMRG calculation is usually substantially
smaller than the total number of occupied and virtual MOs in a molecule, a pre-selection is
necessary before entering the DMRG calculation. We have written a generic wrapper function that
calculates MP2 natural orbitals, selects an orbital subset to be used in the DMRG calculation,
and returns the final active space suggestion. To call this function, nothing more than an SCF
object containing a converged Hartree-Fock result is needed. Moreover, even a simple Hartree-Fock
calculation can be performed automatically.

While the ASF is often a useful tool that greatly simplifies multi-reference calculations, please
be aware of the following difficulties in active space selection:

- Selecting orbitals, especially beyond a highly entangled, minimal active space, is not a
  trivial task.
- Single-orbital entropies are influenced to some degree by the selection of the initial orbital
  space. In particular, the important, entangled partner orbitals need to be included
  in the DMRG calculation.
- Too small bond dimensions in the DMRG calculation can lead to wrong results.

CASSCF or CASCI treat the correlation in a subset of the most important orbitals exactly. However,
they neglect all the remaining electron correlation in the system, which includes the inactive
orbitals. Therefore, CASSCF results tend to be rather of a qualitative than a quantitative nature.
To obtain accurate numbers that can be compared to experiment, it is usually necessary to
use a multi-reference method that includes dynamic correlation, for example NEVPT2 or CASPT2.
The choice of an active space may, to some extent, depend on the specifics of the dynamic
correlation method. For example, spin state energetics calculated with perturbative methods,
such as NEVPT2 or CASPT2, may be improved by using larger active spaces than would be necessary
for a minimal description, thereby also including some dynamic correlation energy in the active
space.

If you find the active space finder a useful tool for your research,
please cite our work.
