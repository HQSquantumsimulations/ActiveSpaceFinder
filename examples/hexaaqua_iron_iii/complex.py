"""High spin / low spin hexaaqua iron (III)."""
from pyscf import gto, scf, fci, mcscf
from pyscf.tools import molden
from asf.wrapper import runasf_from_scf
from asf.utility import pictures_Jmol

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
mol.max_memory = 4500
mol.basis = 'def2-SVP'
mol.charge = 3
mol.spin = 5
mol.verbose = 4
mol.build()

mf = scf.UHF(mol).density_fit()
mf.init_guess = 'atom'
mf.max_cycle = 125
mf.kernel()
mo_new = mf.stability()[0]
while mo_new is not mf.mo_coeff:
    mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
    mo_new = mf.stability()[0]
if (not mf.converged) or (mo_new is not mf.mo_coeff):
    raise Exception('SCF has not converged')

# Use the ASF for lowest quartet and sextet states.
states = [(5, 1), (3, 3)]
act_ele, act_mos, mp2orb = runasf_from_scf(mf, mp2_kwargs={'max_orb': 15}, states=states, maxM=150)

# Save molden file
molden.from_mo(mol, 'mp2_orbitals.molden', mp2orb)

# plot orbitals
pictures_Jmol(mol, mp2orb, act_mos, rotate=(90, 0, 0))

print('Final active space: {0:d} electrons in MOs {1:s}'.format(act_ele, str(act_mos)))

# We want to perform a state-averaged CASSCF calculation, which includes the sextet ground state
# and three triplet states.
# First define the full-CI solvers for the diffecent spin states.
# Sextet
solver5 = fci.FCI(mol)
solver5.spin = 5
solver5.nroots = 1
# Quartet: we want to calculate three states
solver3 = fci.FCI(mol)
solver3.spin = 3
solver3.nroots = 3
# Ensure the the spin does not change
solvers = [fci.addons.fix_spin(s) for s in (solver5, solver3)]

# We use the MP2 natural orbitals that are returned by the wrapper function.
cas_base = mcscf.CASSCF(mol, len(act_mos), act_ele)
weights = (1 / 2, 1 / 6, 1 / 6, 1 / 6)
mc = mcscf.state_average_mix(cas_base, solvers, weights).density_fit()
mo_sorted = mc.sort_mo(act_mos, mo_coeff=mp2orb, base=0)
mc.kernel(mo_coeff=mo_sorted)

# We ordered the solvers such that the sextet comes first and the quartet second.
# mc.e_states contains four entries: the sextet energy, and the three quartet energies.
energy_exc = (mc.e_states[1] - mc.e_states[0]) * 27.2114
print('Excitation energy from sextet to lowest quartet state: {0:.2f} eV'.format(energy_exc))
