# Active space finder

The Active Space Finder (ASF) is a set of functions for the (semi-)automatic
selection of active spaces in molecules. ASF aims to automatize the selection of active orbitals,
thus facilitating the black-box application of multi-reference methods such as CASSCF. The easiest
way to use the ASF is by calling a wrapper function, which executes a generic procedure to
determine the active space with minimal requirements for user intervention. More advanced users
can design their own selection procedure with the individual components
provided by the ASF software.

The code is developed at
[HQS Quantum Simulations](https://quantumsimulations.de), and it is released under the
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0.txt).

This project is partly supported by [PlanQK](https://planqk.de).

ASF requires the following software packages:

- Python3 with numpy and scipy
- [PySCF v.1.7.6](https://github.com/pyscf/pyscf) for general electronic structure calculations
- [BLOCK v.1.5](https://github.com/sanshar/Block) for DMRG calculations
- [Jmol v.14.6.4](http://jmol.sourceforge.net) for plotting

For your convenience, we provide a shell script to install the PySCF and Block software packages,
which only requires an existing conda installation on your computer. It is located in the
directory `build_script`.

## Documentation

Please look [here](https://activespacefinder.readthedocs.io/en/latest) for a documentation of the
ASF, and a tutorial with some example cases.
