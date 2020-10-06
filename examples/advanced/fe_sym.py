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

from pyscf import gto, scf, mp, mcscf, fci
import asf
import numpy as np
ASF = asf.asf()

mol = gto.Mole()
mol.atom = """
Fe -0.0000000 0.0000000 0.0000000
O 0.0000000 2.0622910 0.0000000
H 0.7919274 2.6471973 0.0000000
H -0.7919274 2.6471973 0.0000000
O -0.0000000 0.0000000 2.0622910
H -0.0000000 0.7919274 2.6471973
H 0.0000000 -0.7919274 2.6471973
O 2.0622910 -0.0000000 -0.0000000
H 2.6471973 -0.0000000 0.7919274
H 2.6471973 -0.0000000 -0.7919274
O -0.0000000 -2.0622910 0.0000000
H -0.7919274 -2.6471973 -0.0000000
H 0.7919274 -2.6471973 -0.0000000
O 0.0000000 0.0000000 -2.0622910
H 0.0000000 -0.7919274 -2.6471973
H -0.0000000 0.7919274 -2.6471973
O -2.0622910 0.0000000 0.0000000
H -2.6471973 0.0000000 -0.7919274
H -2.6471973 0.0000000 0.7919274
"""
mol.basis = {'default': 'def2-SVP', 'Fe': 'def2-svpd'}
mol.charge = 3
mol.spin = 3
mol.verbose = 3
mol.symmetry = True
mol.build()
# UHF for UNOs
mf = scf.RHF(mol).density_fit().run(max_cycle=100)
mo_new = mf.stability()[0]
while mo_new is not mf.mo_coeff:
    mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
    mo_new = mf.stability()[0]

# This can be done with the wrapper function as well.

mp2 = mp.MP2(mf).run()
no, occ = asf.nat_orbs_obj(mp2)
sel_mos = ASF.select_mos_occ(occ, machine_limit=12)

mo_list = []

act_ele, act_mos = ASF.find_active_space(mol, asf.orbs_el_n(mf.mo_occ, sel_mos),
                                         sel_mos, no, occup=mf.mo_occ, symm='A')
mo_list.append(act_mos[0])

act_ele, act_mos = ASF.find_active_space(mol, asf.orbs_el_n(mf.mo_occ, sel_mos),
                                         sel_mos, no, occup=mf.mo_occ, symm='B1')
mo_list.append(act_mos[0])

act_ele, act_mos = ASF.find_active_space(mol, asf.orbs_el_n(mf.mo_occ, sel_mos),
                                         sel_mos, no, occup=mf.mo_occ, symm='B2')
mo_list.append(act_mos[0])

act_ele, act_mos = ASF.find_active_space(mol, asf.orbs_el_n(mf.mo_occ, sel_mos),
                                         sel_mos, no, occup=mf.mo_occ, symm='B3')
mo_list.append(act_mos[0])

mol.spin = 5
mol.build()
act_ele, act_mos = ASF.find_active_space(mol, asf.orbs_el_n(mf.mo_occ, sel_mos),
                                         sel_mos, no, occup=mf.mo_occ, symm='A')
mo_list.append(act_mos[0])


act_mos = ASF.select_mroot_AS(mo_list)
act_ele = asf.orbs_el_n(mf.mo_occ, act_mos)

print('Final active space (el, mo)', act_ele, act_mos)

weights = np.ones(5) / 5
solver1 = fci.direct_spin1.FCI(mol)
solver1.spin = 3
solver1 = fci.addons.fix_spin(solver1, ss=3 / 2 * (3 / 2 + 1))
solver1.wfnsym = 'A'
solver1.nroots = 1
solver2 = fci.direct_spin1.FCI(mol)
solver2 = fci.addons.fix_spin(solver2, ss=3 / 2 * (3 / 2 + 1))
solver2.spin = 3
solver2.wfnsym = 'B2'
solver2.nroots = 1
solver3 = fci.direct_spin1.FCI(mol)
solver3 = fci.addons.fix_spin(solver2, ss=3 / 2 * (3 / 2 + 1))
solver3.spin = 3
solver3.wfnsym = 'B3'
solver3.nroots = 1
solver4 = fci.direct_spin1.FCI(mol)
solver4 = fci.addons.fix_spin(solver2, ss=3 / 2 * (3 / 2 + 1))
solver4.spin = 3
solver4.wfnsym = 'B4'
solver4.nroots = 1
solver5 = fci.direct_spin1.FCI(mol)
solver5 = fci.addons.fix_spin(solver2, ss=5 / 2 * (5 / 2 + 1))
solver5.spin = 3
solver5.wfnsym = 'A'
solver5.nroots = 1

# We use the MP2 nat orbs as the entanglement is calculated for them.

mc = mcscf.CASSCF(mf, len(act_mos), act_ele)
mo = mcscf.sort_mo(mc, no, np.array(act_mos) + 1)
mcscf.state_average_mix_(mc, [solver1, solver2, solver3, solver4, solver5], weights)
ener = mc.kernel(mo)

print('Excitation from A to B2: ', (solver1.e_tot - solver2.e_tot) * 27.2114, 'eV')
print('Excitation from A to B3: ', (solver1.e_tot - solver3.e_tot) * 27.2114, 'eV')
print('Excitation from A to B4: ', (solver1.e_tot - solver4.e_tot) * 27.2114, 'eV')
print('Excitation from spin 3 to 5: ', (solver1.e_tot - solver5.e_tot) * 27.2114, 'eV')
