"""Comparing ASF with a manually selected active space for benzene"""
from pyscf import gto, scf, mcscf, mrpt
from pyscf.data.nist import HARTREE2EV

from asf.wrapper import runasf_from_scf
from asf.utility import pictures_Jmol

###################################################################################################
# Molecule setup and RHF calculation.
###################################################################################################
mol = gto.Mole()
mol.atom = 'benzene.xyz'
mol.basis = 'def2-tzvp'
mol.verbose = 4
mol.max_memory = 2500
mol.build()

mf = scf.RHF(mol).density_fit()
mf.kernel()

###################################################################################################
# CASSCF calculation with an automatically selected active space
###################################################################################################

# Get active space of first two singlet states
act_el, act_mos, natorbs = runasf_from_scf(mf, mp2_kwargs={'max_orb': 15}, states=2)

# plot orbitals
pictures_Jmol(mol, natorbs, act_mos, rotate=(90, 0, 0))

# CASSCF with ASF orbitals
# State average CASSCF.
mcASF = mcscf.CASSCF(mf, len(act_mos), act_el).state_average_(weights=(0.5, 0.5))
mcASF.fix_spin()
mo_sorted = mcASF.sort_mo(act_mos, mo_coeff=natorbs, base=0)
mcASF.kernel(mo_coeff=mo_sorted)

# Now, we calculate the NEVPT2 energies of the two electronic states.
# Since NEVPT2 cannot work with state-averaged CASSCF objects, we need to set up a CASCI object.
mciASF = mcscf.CASCI(mf, len(act_mos), act_el)
mciASF.fcisolver.nroots = 2
mciASF.fix_spin()
# Calculate the CASCI energies
e1 = mciASF.kernel(mo_coeff=mcASF.mo_coeff)
# Calculate the NEVPT2 correlation energies for both roots separately.
e_corrASF0 = mrpt.NEVPT(mciASF, root=0).kernel()
e_corrASF1 = mrpt.NEVPT(mciASF, root=1).kernel()

###################################################################################################
# CASSCF calculation with a manually selected active space
###################################################################################################

# The hand-picked (6, 6) active space with canonical RHF orbitals.
orb_manual = [16, 19, 20, 21, 22, 30]
# State average CASSCF.
mc_manual = mcscf.CASSCF(mf, len(orb_manual), act_el).state_average_(weights=(0.5, 0.5))
mc_manual.fix_spin()
mo_sorted = mc_manual.sort_mo(orb_manual, mo_coeff=mf.mo_coeff, base=0)
mc_manual.kernel(mo_coeff=mo_sorted)

# Calculate the NEVPT2 energies of the two electronic states.
# As before, we need to set up a new CASCI object.
mci_manual = mcscf.CASCI(mf, len(orb_manual), act_el)
mci_manual.fcisolver.nroots = 2
mci_manual.fix_spin()
# Calculate the CASCI energies.
e2 = mci_manual.kernel(mo_coeff=mc_manual.mo_coeff)
# Calculate the NEVPT2 correlation energies for both roots separately.
e_corr_man0 = mrpt.NEVPT(mci_manual, root=0).kernel()
e_corr_man1 = mrpt.NEVPT(mci_manual, root=1).kernel()

###################################################################################################
# Print the results: excitation energies computed with CASSCF and NEVPT2.
###################################################################################################
CAS_ex = (e1[0][1] - e1[0][0]) * HARTREE2EV
PT_ex = CAS_ex + (e_corrASF1 - e_corrASF0) * HARTREE2EV
print('CASSCF excitation energy with automatic selection: {0:.3f} eV'.format(CAS_ex))
print('NEVPT2 excitation energy with automatic selection: {0:.3f} eV'.format(PT_ex))
CAS_ex = (e2[0][1] - e2[0][0]) * HARTREE2EV
PT_ex = CAS_ex + (e_corr_man1 - e_corr_man0) * HARTREE2EV
print('CASSCF excitation energy with manual selection: {0:.3f} eV'.format(CAS_ex))
print('NEVPT2 excitation energy with manual selection: {0:.3f} eV'.format(PT_ex))
