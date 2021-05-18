from math import sqrt

import numpy as np

from pyscf.gto import Mole
from pyscf.lib import logger
from pyscf.scf import RHF
from pyscf.tools import molden

from asf import ASFCI

################################################################################
# Construct RHF orbitals for the H2 molecule at its equilibrium geometry.
################################################################################

mol = Mole()
mol.atom = '''
H 0.00 0.00 0.00
H 0.00 0.00 0.74
'''
mol.spin = 0
mol.charge = 0
mol.basis = '6-31G'
mol.verbose = logger.INFO
mol.build()

mf = RHF(mol)
mf.kernel()
molden.from_scf(mf, 'H2_RHF.molden')

################################################################################
# Rotate the bonding and antibonding orbital by 45 degrees
################################################################################
U = 1.0 / sqrt(2) * np.array([[1, 1], [-1, 1]])
mo_rot = mf.mo_coeff.copy()
mo_rot[:, :2] = mf.mo_coeff[:, :2].dot(U)

molden.from_mo(mol, 'H2_rotated.molden', mo_rot)

################################################################################
# Illustrate orbital selection with the RHF orbitals and the rotated orbitals.
################################################################################
print()
print('-' * 80)
print('Active space selection with RHF orbitals')
print('-' * 80)
print()
sf = ASFCI(mol, mf.mo_coeff)
sf.runCI()
sf.selectOrbitals()

print()
print('-' * 80)
print('Active space selection with rotated orbitals')
print('-' * 80)
print()
sf = ASFCI(mol, mo_rot)
sf.runCI()
sf.selectOrbitals()
