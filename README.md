# Active space finder

## Requirements

The code is built upon [PySCF 1.7.x](https://github.com/pyscf/pyscf) and [Block 1.5](https://github.com/sanshar/StackBlock) for DMRG calculations.

## Directory structure

[asf](asf) contains the active space finder code.

[examples](examples) contains several examples that illustrate the usage of the code.

## Finding active spaces for individual molecules

Important functions and classes are listed in [\_\_init\_\_.py](asf/__init__.py) and can be imported from the `asf` directory. Example:

```
from asf import ASFDMRG
```

### The `ASFCI` class

The `ASFCI` class in [casci.py](asf/casci.py) contains much of the necessary functionality to perform an entropy-based orbital selection with normal CASCI rather than DMRG-CASCI.

It is initialized with a PySCF molecule object, a matrix of MO coefficients, and the number of electrons and orbitals in the initial active space. If the latter two are omitted, a full CI calculation is assumed by default. A more fine-grained initial active space selection can be provided through the `setMOList` method, which overrides any settings in the constructor.

```
from asf import ASFCI

# CI-based active space finder with 7 electrons in 9 orbitals as the initial subspace.
sf = ASFCI(mol, mo_coeff, 7, 9)

# OR: Initialize a CI-based active space finder.
sf = ASFCI(mol, mo_coeff)
# Define an initial subspace of 10 electrons in the given list of orbitals.
sf.setMOList(10, [4, 7, 8, 10, 13, 14, 15, 16, 19, 20, 22])
```

After the active space finder has been set up, the active space can be selected through a CASCI calculation:

```
sf.runCI()
nel, mo_list = sf.selectOrbitals()
```

The `selectOrbitals` method accepts several options, including on thresholds and whether to use cumulant pair information.

### The `ASFDMRG` class

The `ASFDMRG` class in [dmrg.py](asf/dmrg.py) derives from the `ASFCI` class and contains the functionality needed to perform a selection based on entropies from a DMRG calculation.

The actual DMRG calculation needs to be called through the `runDMRG` method, which accepts DMRG-specific options, instead of `runCI`:

```
from asf import ASFDMRG

sf = ASFDMRG(...)
sf.setMOList(...)
sf.runDMRG(maxM=250)
nel, mo_list = sf.selectOrbitals()
```

### Calculating natural orbitals

For any molecule of a realistic size, a subset of orbitals needs to be selected prior to performing the DMRG calculation. This is within the responsibility of the user.

A useful choice to this end are natural orbitals of a correlated density, e.g. from MP2 or CCSD, or the natural orbitals of an unrestricted HF or DFT density. [natorbs.py](asf/natorbs.py) provides some functions to help with the construction and selection of natural orbitals. 

```
from pyscf.scf import UHF
from pyscf.mp import MP2
from pyscf.cc import CCSD

from asf import UHFNaturalOrbitals, MP2NaturalOrbitals, CCSDNaturalOrbitals
from asf import selectNaturalOccupations

# Perform a UHF calculation with a molecule that was defined previously
mf = UHF(mol)

# UHF natural orbitals
occupation_numbers, natural_orbitals = UHFNaturalOrbitals(mf)

# MP2 natural orbitals
pt = MP2(mf)
pt.kernel()
occupation_numbers, natural_orbitals = MP2NaturalOrbitals(pt)

# CCSD natural orbitals
cc = CCSD(mf)
cc.kernel()
occupation_numbers, natural_orbitals = CCSDNaturalOrbitals(pt)

# Perform a selection based on natural orbitals
nel, mo_list = selectNaturalOccupations(occupation_numbers)

# Use different truncation thresholds, but limit the number of orbitals to 30 at most.
nel, mo_list = selectNaturalOccupations(occupation_numbers, lower=0.01, upper=1.99, max_orb=30)
```

## Examples

### Singlet oxygen

[O2/singlet.py](examples/O2/singlet.py) is a simple example based on the lowest singlet state of molecular oxygen. It first calculates a set of SCF orbitals, albeit for the triplet state, as it is easier to converge. Subsequently, orbitals are selected based on entropies obtained from a DMRG-CASCI calculation.

### Ozone ground state

[O3/O3.py](examples/O3/O3.py) illustrates active space selection for the singlet state of ozone with different sets of orbitals: SCF orbitals, unrestricted natural orbitals, MP2 natural orbitals on top of a broken-symmetry wave function, and CASSCF orbitals calculated with a (2, 2) minimal active space.

### Orbital rotation in the hydrogen molecule

Single-orbital entropies can be taken as an indicator of multi-reference character in a system. However, since they depend strongly on the choice of orbitals, this is not a reliable measure in general.

[H2/H2.py](examples/H2/H2.py) illustrates this point by calculating entropies twice: first, using SCF orbitals obtained for the H2 molecule at equilibrium bond distance; and second, for the same molecule after mixing the HOMO and the LUMO in equal parts.

With the RHF orbitals, small entropies are obtained (as would be expected). After rotating the HOMO and LUMO to what are effectively orthogonalized AOs, the entropies are close to the largest possible value of 1.39. The entropy measure cannot be used to distinguish between orbital rotations, which are similar to single excitations on a reference determinant on the one hand; and genuine correlation, which requires higher-order excitations, on the other hand.
