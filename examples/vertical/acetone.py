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
#########################################
#   HQS ASF example
#########################################
#   Vertical Excitation of dicarbon
#########################################
#   Reza shirazi
#########################################


#################################
#    Libraries
#################################
import asf
from asf import color
import os
from pyscf import gto, scf, mcscf, mrpt
import warnings
import numpy as np

# Silence errors
warnings.simplefilter("ignore")

# Title
# Used for saving the SCF in a chk file
name = 'acetone'
final_spin = 0
NThreads = 8

# remove old log file
if os.path.isfile('asf.log'):
    os.remove('asf.log')

# Set up the PySCF molecule
mol = gto.Mole()
mol.atom = """
O 0.000 0.000 1.405
C 0.000 0.000 0.185
C 0.000 -1.287 -0.616
C 0.000 1.287 -0.616
H 0.000 -2.145 0.055
H 0.000 2.145 0.055
H 0.882 -1.321 -1.265
H -0.882 -1.321 -1.265
H 0.882 1.321 -1.265
H -0.882 1.321 -1.265
"""
mol.basis = {'default': 'def2-svp', 'H': 'sto-3g'}
mol.charge = 0
mol.spin = 2
mol.verbose = 2
mol.max_memory = 6000
mol.build()

mf = scf.UHF(mol).density_fit()
mf.conv_tol = 1e-10

if os.path.isfile(name + '.chk'):
    mf.chkfile = name + '.chk'
    mf.init_guess = 'chkfile'

mf.run(max_cycle=100)
mo_new = mf.stability()[0]
while mo_new is not mf.mo_coeff:
    mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
    mo_new = mf.stability()[0]
mf.newton().kernel(dm0=mf.make_rdm1())


ASF = asf.asf()

ele, mos = ASF.fas_no_guess(mf, nat_type='MP2', machine_limit=10, nroots=2, spin=0, converge_=True,
                            ent_conv=1e-6)

print("")
for i in range(len(mos)):
    print(color.BOLD + 'Suggested active space for root:', i, " " + color.END, color.RED,
          ele[i], color.END, color.BOLD + 'electrons in these MOs:' + color.END, color.RED,
          len(mos[i]), ": ", mos[i], color.END)
print("")

# If state average calculations are done Check the active spaces
mos = ASF.select_mroot_AS(mos)
ele = asf.orbs_el_n(mf.mo_occ, mos)

print(color.BOLD + 'Selected active space' + color.END, color.RED, ele, color.END,
      color.BOLD + 'electrons in these MOs:' + color.END, color.RED, len(mos), ": ",
      mos, color.END)
print("")

# CASSCF on DMRG nat orbs
print(color.BOLD + "State average CASSCF calculations" + color.END)
mc = mcscf.CASSCF(mf, len(mos), ele)
mo = mcscf.sort_mo(mc, ASF.init_nat, np.array(mos) + 1)

mc = mc.state_average([0.5, 0.5])
mc.fcisolver.spin = final_spin
mc.fix_spin(ss=final_spin)
mc.kernel(mo)

# Save info
cas_orbs = mc.mo_coeff


mc = mcscf.CASCI(mf, len(mos), ele)
mc.fcisolver.nroots = 2
mc.fix_spin(ss=final_spin)

ener = mc.kernel(cas_orbs)
print(color.BOLD + 'Root [0]:' + color.END, color.RED, ener[0][0], color.END,
      color.BOLD + 'Root [1]:' + color.END, color.RED, ener[0][1], color.END)

print(color.BOLD + "NEVPT2 calculations..." + color.END)
NEVPT2_H = mrpt.NEVPT(mc, root=1).kernel()
print(color.BOLD + 'NEVPT2 ROOT[1]:' + color.END, color.RED, NEVPT2_H, color.END)

NEVPT2_L = mrpt.NEVPT(mc, root=0).kernel()
print(color.BOLD + 'NEVPT2 ROOT[0]:' + color.END, color.RED, NEVPT2_L, color.END)

print(color.BOLD + "Excited state calculations..." + color.END)
CAS_ex = (ener[0][1] - ener[0][0]) * 27.2114
NEVPT_corr = (NEVPT2_H - NEVPT2_L) * 27.2114
print("")

print(color.BOLD + 'CASCF excitation (eV):' + color.END, color.RED, '{0:.4f}'.format(CAS_ex),
      color.END, color.BOLD + 'NEVPT2 excitation (eV):' + color.END, color.RED,
      '{0:.4f}'.format(NEVPT_corr + CAS_ex), color.END)
