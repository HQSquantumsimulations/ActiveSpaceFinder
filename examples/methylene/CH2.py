"""Basic active space selection for methylene."""
from pyscf import gto, mcscf
from asf.wrapper import runasf_from_mole
from asf.utility import pictures_Jmol

# build the mol object
mol = gto.Mole()
mol.atom = 'ch2.xyz'
mol.basis = 'def2-tzvp'
mol.spin = 2
mol.verbose = 4
mol.max_memory = 2500
mol.build()

# Get active space of first two singlet states
nel, mo_list, mo_coeff = runasf_from_mole(mol, states=[(0, 1), (2, 1)], entropy_threshold=0.2,
                                          switch_dmrg=4)

# plot the valence orbitals
pictures_Jmol(mol, mo_coeff, mo_list=[1, 2, 3, 4, 5, 6], rotate=(45, 0, 0))

# CASSCF with ASF orbitals for the singlet and the triplet
for spin in 0, 2:
    print('\nCASSCF with spin = {0:d}\n'.format(spin))
    mc = mcscf.CASSCF(mol, len(mo_list), nel)
    mc.fcisolver.spin = spin
    mc.fix_spin()
    mo_sorted = mc.sort_mo(mo_list, mo_coeff=mo_coeff, base=0)
    mc.kernel(mo_coeff=mo_sorted)
