"""Basic active space selection: pi orbitals of ethene"""

from pyscf import gto, mcscf
from pyscf.lib import logger

from asf.wrapper import runasf_from_mole
from asf.utility import pictures_Jmol

# Set up the molecule.
mol = gto.Mole()
mol.atom = 'c2h4.xyz'
mol.basis = 'def2-SVP'
mol.spin = 0
mol.charge = 0
mol.verbose = logger.INFO
mol.build()

# Wrapper function to:
#
# 1) Perform an SCF calculation
# 2) Calculate MP2 natural orbitals.
# 3) Run CASCI in a subset of MP2 natural orbitals.
# 4) Perform the orbital selection.
#
# We get:
# nel:      Number of electrons in the selected active space.
# mo_list:  List of orbitals making up the active space.
# mo_coeff: The matrix containing the MO coefficients (active and inactive).
#           Here, the MO coefficients are MP2 natural orbitals. The rows of the matrix index AOs,
#           the columns index MOs.
nel, mo_list, mo_coeff = runasf_from_mole(mol)

# Plot the orbitals.
pictures_Jmol(mol, mo_coeff, mo_list=mo_list, rotate=(-45, 0, 0))

# Now we set up a CASSCF calculation.
# ncas: number of active orbitals
ncas = len(mo_list)
mc = mcscf.CASSCF(mol, ncas, nel)

# Generally, we need to sort the MOs such that the columns are ordered from left to right:
# 1) occupied inactive
# 2) active
# 3) unoccupied inactive
# Note that python counts from 0 by default, but the default value for base in sort_mo is 1.
# Therefore, we need to set base=0 explicitly.
mo_sorted = mc.sort_mo(mo_list, mo_coeff=mo_coeff, base=0)

# Run the CASSCF calculation.
mc.kernel(mo_coeff=mo_sorted)
