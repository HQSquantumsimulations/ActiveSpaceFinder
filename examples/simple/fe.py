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
mol.basis = {'default': 'minao'}
mol.charge = 3
mol.spin = 3
mol.verbose = 0
mol.build()
# UHF for UNOs
mf = scf.RHF(mol).run(max_cycle=100)
mo_new = mf.stability()[0]
while mo_new is not mf.mo_coeff:
    mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
    mo_new = mf.stability()[0]

# Call the wrapper function.
ASF = asf.asf()

ele, mos = ASF.fas_no_guess(mf, nat_type='MP2', machine_limit=11)
