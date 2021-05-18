from pyscf.gto import Mole
from pyscf.scf import UHF
from pyscf.mp import MP2
from pyscf.lib import logger
from pyscf.tools import molden
from pyscf.mcscf import CASSCF

from asf import UHFNaturalOrbitals, MP2NaturalOrbitals, selectNaturalOccupations, ASFDMRG, ASFCI

# set up the singlet ozone molecule
mol = Mole()
mol.atom = '''
O   0.000   0.000   0.000
O   0.670   1.089   0.000
O   0.670  -1.089   0.000
'''
mol.basis = 'def2-SVP'
mol.charge = 0
mol.spin = 0
mol.verbose = logger.INFO
mol.build()

################################################################################
# Preliminary calculations to compute different types of orbitals
################################################################################

# run a broken-symmetry UHF calculation
# use stability analysis as a hack to break the spin-symmetry
mf = UHF(mol)
mf.kernel()
mo_new = mf.stability()[0]
dm_new = mf.make_rdm1(mo_coeff=mo_new)
mf.kernel(dm0=dm_new)
# calculate unrestricted natural orbitals (UNOs)
natocc_uhf, natorb_uhf = UHFNaturalOrbitals(mf)
molden.from_mo(mol, 'uhfnat.molden', natorb_uhf, occ=natocc_uhf)

# calculate UHF-MP2 natural orbitals
pt = MP2(mf)
pt.kernel()
natocc_mp2, natorb_mp2 = MP2NaturalOrbitals(pt)
molden.from_mo(mol, 'ump2nat.molden', natorb_mp2, occ=natocc_mp2)

# CASSCF calculation with a (2, 2) minimal active space
cas22 = CASSCF(mol, 2, 2)
cas22.fix_spin()
cas22.kernel(mo_coeff=natorb_uhf)
molden.from_mcscf(cas22, 'cas22.molden')

################################################################################
# Apply the active space finder with different types of orbitals
################################################################################

# First we try the active space finder for the UHF alpha orbitals.
# We "cheat" by selecting a (12, 9) initial space, corresponding to MOs from 2p orbitals.
print('')
print('DMRG with UHF orbitals')
sf_uhf = ASFDMRG(mol, mf.mo_coeff[0], 12, 9)
sf_uhf.runDMRG(maxM=200)
nel_uhf, molist_uhf = sf_uhf.selectOrbitals()
print('Selected {0:d} electrons in alpha orbitals {1}.'.format(nel_uhf, molist_uhf))
print('')

# Try the active space finder with UHF natural orbitals.
# Use CI instead of DMRG as only two orbitals make it past the natural orbital selection criterion.
print('')
print('DMRG with unrestricted HF natural orbitals')
sf_uno = ASFCI(mol, natorb_uhf)
sf_uno.setMOList(*selectNaturalOccupations(natocc_uhf))
sf_uno.runCI()
nel_uno, molist_uno = sf_uno.selectOrbitals()
print('Selected {0:d} electrons in UNOs {1}.'.format(nel_uno, molist_uno))
print('')

# Try the active space finder with MP2 natural orbitals.
# Arguably the most honest calculation in this file:
# prescreen with the natural orbital selection criterion, then select the final active space.
print('')
print('DMRG with unrestricted MP2 natural orbitals')
sf_mp2 = ASFDMRG(mol, natorb_mp2)
init_nel_mp2, init_molist_mp2 = selectNaturalOccupations(natocc_mp2)
sf_mp2.setMOList(init_nel_mp2, init_molist_mp2)
sf_mp2.runDMRG(maxM=200)
nel_mp2, molist_mp2 = sf_mp2.selectOrbitals()
print('Selected {0:d} electrons in MP2 natural orbitals {1}.'.format(nel_mp2, molist_mp2))
print('')

# Finally, try the active space finder with the orbitals from a converged CASSCF(2, 2) calculation.
# We "cheat" by pre-selecting p-type valence orbitals and manually changing the threshold.
print('')
print('DMRG with CASSCF(2,2) natural orbitals')
sf_cas = ASFDMRG(mol, cas22.mo_coeff, 12, 9)
sf_cas.runDMRG(maxM=200)
nel_cas, molist_cas = sf_cas.selectOrbitals(threshold=0.1)
print('Selected {0:d} electrons in orbitals {1}.'.format(nel_cas, molist_cas))
print('')
