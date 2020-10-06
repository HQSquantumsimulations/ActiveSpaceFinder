# Copyright 2020 HQS Quantum Simulations GmbH
# Reza Ghafarian Shirazi, Thilo Mast.
# reza.shirazi@quantumsimulations.de
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pyscf import scf, mcscf, mp, ci, gto
import asf
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

no, occ = asf.UNOs(mf)
print('Unretricted natural orbitals\n', occ)
print('Total occupation of unrestricted natural orbitals:', sum(occ), '\n')

# UHF UMP2
mp2 = mp.UMP2(mf).run()
no, occ = asf.nat_orbs_obj(mp2)
print('MP2 natural orbitals\n', occ)
print('Total occupation of MP2 natural orbitals:', sum(occ), '\n')

# CISD
mci = ci.UCISD(mf).run()
no, occ = asf.nat_orbs_obj(mci)
print('CISD natural orbitals\n', occ)
print('Total occupation of CISD natural orbitals:', sum(occ), '\n')

# CASSCF
mc = mcscf.CASSCF(mf, 6, 6)
mc.kernel()
S = mc.mol.intor_symmetric('int1e_ovlp')
no, occ = asf.nat_orbs_rdm1AO_rhf(mc.make_rdm1(), mc.mo_coeff, mc.mol)

no_active_1 = no[:, 3:9]
print('CASCF(6,6) natural orbitals\n', occ)
print('Total occupation of CASSCF(6,6) natural orbitals:', sum(occ))

# CASSCF
mc = mcscf.CASSCF(mf, 6, 6)
mc.kernel()
S = mc.mol.intor_symmetric('int1e_ovlp')
no, occ = asf.nat_orbs_rdm1AO_rhf(mc.make_rdm1(), mc.mo_coeff, mc.mol, mode=2)
print('CASCF(6,6) natural orbitals\n', occ)
print('Total occupation of CASSCF(6,6) natural orbitals:', sum(occ))

# Only active natural orbitals are meaningful
no_active_2 = no[:, 3:9]

if np.allclose(abs(no_active_1), abs(no_active_2)):
    print('Natural orbitals calculated from both modes match')
else:
    raise Exception('A mistake found')
