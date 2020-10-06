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

from pyscf import gto, scf
import asf
import numpy as np
from pyscf.mcscf import avas

ASF = asf.asf()

mol = gto.Mole()
mol.atom = """
C          0.00000        0.00000        0.00000
C          0.00000        0.00000        1.20000
"""
mol.basis = 'def2-svp'
mol.charge = 0
mol.spin = 0
mol.build()
# UHF for UNOs
mf = scf.RHF(mol).run(max_cycle=100)
mo_new = mf.stability()[0]
while mo_new is not mf.mo_coeff:
    mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
    mo_new = mf.stability()[0]

# AVAS initial guess.
ao_labels = ['C 2s', 'C 2p']
norb, ne_act, orbs = avas.avas(mf, ao_labels, canonicalize=False)

# Plot AVAS selected orbitals.
# orbital list
fao = asf.act_fao(mf.mo_occ, ne_act)
orblist = fao + np.array(list(range(norb)))
asf.visualize_mos(mol, orbs, orblist)

# Select an active space using entropies.
ele, mos = ASF.find_active_space(mol, ne_act, norb, orbs, plot=True)
