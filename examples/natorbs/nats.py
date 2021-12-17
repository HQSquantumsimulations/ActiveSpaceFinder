"""Example for calculating natural orbitals."""
from pyscf import scf, mcscf, mp, cc, gto
from asf import natorbs
import numpy as np

mol = gto.M(atom='C 0 0 0; C 0 0 1.24253', basis='def2-svp', spin=2)
mol.verbose = 0

# UHF for UNOs
mf = scf.UHF(mol).run(max_cycle=100)
mo_new = mf.stability()[0]
while mo_new is not mf.mo_coeff:
    mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
    mo_new = mf.stability()[0]

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

occ, no = natorbs.UHFNaturalOrbitals(mf)
print('Unretricted natural orbitals\n', occ)
print('Total occupation of unrestricted natural orbitals:', sum(occ), '\n')

# UHF UMP2
mp2 = mp.UMP2(mf).run()
occ, no = natorbs.MP2NaturalOrbitals(mp2)
print('MP2 natural orbitals\n', occ)
print('Total occupation of MP2 natural orbitals:', sum(occ), '\n')

# CCSD
mci = cc.UCCSD(mf).run()
occ, no = natorbs.CCSDNaturalOrbitals(mci)
print('CCSD natural orbitals\n', occ)
print('Total occupation of CISD natural orbitals:', sum(occ), '\n')

# CASSCF
mc = mcscf.CASSCF(mf, 6, 6).run()
occ, no = natorbs.AOnaturalOrbitals(mc.make_rdm1(), mc.mol.intor_symmetric('int1e_ovlp'))
print('CASCF(6,6) natural orbitals\n', occ)
print('Total occupation of CASSCF(6,6) natural orbitals:', sum(occ))
