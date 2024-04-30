# Active Space Finder

The Active Space Finder (ASF) is a set of functions for the (semi-)automatic selection of active spaces in molecules to be employed with methods such as CASSCF. Choosing an appropriate set of active orbitals can be a complicated task, which requires a significant amount of expertise. We have set out to develop a tool to make such calculations easier and more accessible to both expert and non-expert users.

Employing early-stage quantum devices, in the so-called Noisy-Intermediate-Scale-Quantum (NISQ) era, for applications in quantum chemistry requires the majority of problems to be reduced to their most important degrees of freedom. The ASF sets out to determine those parts of a molecular system that need to be treated at the most advanced quantum mechanical level.

## Installation

#### Local installation

We recommend Python versions `3.9` to `3.11`. Please note that `block2=0.5.2` has not been packaged
for Python newer than `3.11` on [pypi.org](https://pypi.org/project/block2/).

A local installation of the ASF package from the source directory requires two steps:
```sh
pip install .
./init_dmrgscf_settings.sh
```
First, the dependencies are installed via `pip`. Because the ASF code is built upon [PySCF](https://github.com/pyscf/pyscf) and [Block2](https://github.com/block-hczhai/block2-preview) for DMRG calculations, a configuration file for PySCF's `dmrgscf` plugin is created in a second step. For convenience, the ASF package includes a script to create the configuration file automatically (run `./init_dmrgscf_settings.sh --help` to learn more about its usage).

## Quick Start

Selecting an active space for a single molecule structure with the help of the ASF usually
includes the following steps:

1. Calculating an SCF solution for the molecule. We recommend a UHF calculation, even for singlet
   states, as a broken-symmetry solution helps with the active space selection. The ASF package
   includes a convenience function (`asf.scf.stable_scf`) that performs a stability analysis and
   reruns an SCF calculation if required.
2. Performing a perturbative calculation to obtain an initial orbital set. Most importantly, this
   is necessary to select a suitable number of orbitals for the subsequent DMRG calculation.
   Moreover, (UHF-)MP2 natural orbitals tend to be closer to converged CASSCF active orbitals than
   the canonical MOs from Hartree-Fock.
3. A fast, but low-accuracy DMRG calculation is performed for the initial orbital set to obtain
   refined correlation information. It is used to calculate information such as the two-electron
   cumulant and one-orbital entropies that lead to the final active space selection.
4. This information is analyzed automatically to obtain active space suggestions for CASSCF with
   desirable features such as inclusion of correlation partner orbitals. A strength of the ASF
   is its ability to provide the user with multiple active space suggestions for a single molecule.

The easiest way to start is to use one of the wrapper functions that execute these steps on their
own. First, it is necessary to define the molecule as a PySCF object.

```python
from pyscf.gto import Mole
from pyscf.lib import logger
from asf.wrapper import find_from_mol, sized_space_from_mol

ethene_geometry = """
    C        0.6695      0.0000      0.0000
    C       -0.6695      0.0000      0.0000
    H        1.2320      0.0000     -0.9289
    H        1.2320      0.0000      0.9289
    H       -1.2320      0.0000      0.9289
    H       -1.2320      0.0000     -0.9289
"""
mol = Mole(atom=ethene_geometry, basis="def2-SVP", charge=0, spin=0, verbose=logger.INFO).build()
```

Using a molecule definition, the function `find_from_mol` determines one active space using an
entropy threshold and the pair information - for ethene this will result in a (2, 2) space.

```python
# Find one active space.
active_space = find_from_mol(mol)
```

Likewise, it is possible to request a specific active space size to be found:

```python
# Determine an active space with four orbitals.
active_space = sized_space_from_mol(mol, size=4)
```

The returned object contains the number of active electrons, the indices of the active orbitals
and the MO coefficient matrix that these indices refer to. It is important to use the returned MO
coefficient matrix for active space calculations - one should not attempt to combine the list of MO
indices with orbitals originating from an SCF calculation or elsewhere!

For worked examples with detailed instructions, please check the [examples](./examples)
directory.

## Documentation

Please [click here](https://hqsquantumsimulations.github.io/ActiveSpaceFinder) for the
Active Space Finder documentation.

Sphinx documentation of the package is contained in the `doc` folder. It can be built
after checking out the repository with the following steps. The documentation integrates Jupyter
notebooks that require [Jmol](https://jmol.sourceforge.net/) to create orbital plots. Building the
full documentation, therefore, requires a Jmol installation. Note that Jmol can be installed via
the conda package manager, too.

```shell
# Installation of the ASF including the documentation dependencies
pip install .[doc]
./init_dmrgscf_settings.sh
# Building the documentation.
cd doc
make html
```

Building the full documentation can take several minutes due to the execution of Jupyter notebooks.
Afterwards, the built documentation is contained under `doc/_build/html/index.html`.

## Funding

This work was partly supported by the German Federal Ministry for Economic Affairs and Climate
Action through project "PlanQK" (01MK20005H) and by the German Federal Ministry of Education and
Research through projects "MANIQU" (13N15576) and "PhoQuant" (13N16107).
