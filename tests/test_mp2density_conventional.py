# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
from math import isclose

import numpy as np
from pyscf import ao2mo
from pyscf.mp import MP2

from asf import mp2density_conventional
from asf.natorbs import unrestricted_natural_orbitals


def test_make_rdm1_rmp2(ammonia_RHF):
    """
    Testing the modified MP2 1-RDM with RHF reference against UHF reference.
    """
    rmf = ammonia_RHF
    # Create UHF object directly from RHF (without running SCF).
    umf = rmf.to_uhf()

    # Run MP2 for RHF and UHF references.
    rpt = MP2(rmf).run()
    upt = MP2(umf).run()

    # Calculate MP2 1-RDM with RHF and UHF references.
    dm1 = mp2density_conventional.make_rdm1_rmp2(rpt, rmf.mo_energy)
    dm1a, dm1b = mp2density_conventional.make_rdm1_ump2(upt, umf.mo_energy)

    # Sanity check: alpha and beta parts of UHF-based 1-RDM must be equal.
    assert np.allclose(dm1a, dm1b, atol=1e-12, rtol=0)
    # Actual test: check the RHF-based 1-RDM equals the UHF-based 1-RDM.
    assert np.allclose(dm1a + dm1b, dm1, atol=1e-12, rtol=0)


def test_make_rdm2_rmp2(ammonia_RHF):
    """
    Testing the modified MP2 2-RDM with RHF reference against UHF reference.
    """
    rmf = ammonia_RHF
    # Create UHF object directly from RHF (without running SCF).
    umf = rmf.to_uhf()

    # Run MP2 for RHF and UHF references.
    rpt = MP2(rmf).run()
    upt = MP2(umf).run()

    # Calculate MP2 1-RDM with RHF and UHF references.
    dm2 = mp2density_conventional.make_rdm2_rmp2(rpt, rmf.mo_energy)
    dm2aa, dm2ab, dm2bb = mp2density_conventional.make_rdm2_ump2(upt, umf.mo_energy)

    # Sanity check: pure alpha and beta parts of UHF-based 2-RDM must be equal.
    assert np.allclose(dm2aa, dm2bb, atol=1e-12, rtol=0)
    # Actual test: check the RHF-based 2-RDM equals the UHF-based 2-RDM.
    dm2_ref = dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb
    assert np.allclose(dm2, dm2_ref, atol=1e-12, rtol=0)


def test_make_rdm1_ump2(OH_radical_UMP2):
    """
    Checking the modified MP2 1-RDM with UHF reference.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object

    # Calculate the 1-RDM (with alpha and beta parts).
    rdm1a, rdm1b = mp2density_conventional.make_rdm1_ump2(OH_radical_UMP2, mf.mo_energy)

    # Density matrix must be symmetric.
    assert np.allclose(rdm1a, rdm1a.T, atol=1e-14, rtol=0)
    assert np.allclose(rdm1b, rdm1b.T, atol=1e-14, rtol=0)
    # Trace of the density matrix must equal the number of electrons (5 spin-up, 4 spin-down).
    assert isclose(np.trace(rdm1a), 5, abs_tol=1e-12, rel_tol=0)
    assert isclose(np.trace(rdm1b), 4, abs_tol=1e-12, rel_tol=0)

    # Checking a few natural occupation numbers (eigenvalues) calculated with this implementation.
    S = mf.get_ovlp()
    natocc, _ = unrestricted_natural_orbitals((rdm1a, rdm1b), mf.mo_coeff, S)
    assert isclose(natocc[3], 1.97213554, abs_tol=1e-6, rel_tol=0)
    assert isclose(natocc[4], 0.99367890, abs_tol=1e-6, rel_tol=0)
    assert isclose(natocc[5], 0.02106954, abs_tol=1e-6, rel_tol=0)


def test_make_rdm2_ump2(OH_radical_UMP2):
    """
    Checking consistency of the modified MP2 2-RDM (UHF reference) with 1-RDM.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object

    # Calculate 1-RDM.
    rdm1a, rdm1b = mp2density_conventional.make_rdm1_ump2(OH_radical_UMP2, mf.mo_energy)
    # Calculate 2-RDM.
    rdm2aa, rdm2ab, rdm2bb = mp2density_conventional.make_rdm2_ump2(OH_radical_UMP2, mf.mo_energy)

    # Check index permutation symmetries of the 2-RDM (pure alpha part).
    assert np.allclose(rdm2aa, -rdm2aa.transpose((2, 1, 0, 3)), atol=1e-14, rtol=0)
    assert np.allclose(rdm2aa, -rdm2aa.transpose((0, 3, 2, 1)), atol=1e-14, rtol=0)
    assert np.allclose(rdm2aa, rdm2aa.transpose((1, 0, 3, 2)), atol=1e-14, rtol=0)

    # Check index permutation symmetry of the 2-RDM (mixed alpha/beta part).
    assert np.allclose(rdm2ab, rdm2ab.transpose((1, 0, 3, 2)), atol=1e-14, rtol=0)

    # Check index permutation symmetries of the 2-RDM (pure beta part).
    assert np.allclose(rdm2bb, -rdm2bb.transpose((2, 1, 0, 3)), atol=1e-14, rtol=0)
    assert np.allclose(rdm2bb, -rdm2bb.transpose((0, 3, 2, 1)), atol=1e-14, rtol=0)
    assert np.allclose(rdm2bb, rdm2bb.transpose((1, 0, 3, 2)), atol=1e-14, rtol=0)

    # Checking that summing over one electron yields the 1-RDM.
    nocca, noccb = OH_radical_UMP2.get_nocc()
    # sum_r rdm2aa[p, q, r, r] == (N(alpha) - 1) * rdm2a[p, q]
    assert np.allclose(np.einsum("pqrr->pq", rdm2aa), (nocca - 1) * rdm1a, atol=1e-12, rtol=0)
    # sum_r rdm2ab[p, q, r, r] == N(beta) * rdm2a[p, q]
    assert np.allclose(np.einsum("pqrr->pq", rdm2ab), noccb * rdm1a, atol=1e-12, rtol=0)
    # sum_r rdm2bb[p, q, r, r] == (N(beta) - 1) * rdm2b[p, q]
    assert np.allclose(np.einsum("pqrr->pq", rdm2bb), (noccb - 1) * rdm1b, atol=1e-12, rtol=0)
    # sum_r rdm2ab[r, r, p, q] == N(alpha) * rdm2b[p, q]
    assert np.allclose(np.einsum("rrpq->pq", rdm2ab), nocca * rdm1b, atol=1e-12, rtol=0)


def test_rdm_vs_mp3(OH_radical_UMP2):
    """
    Checking the consistency of the modified MP2 RDMs with MP3 energy contributions.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object
    mol = mf.mol

    # Alpha and beta Fock matrix in canonical MO basis.
    Fa = np.diag(mf.mo_energy[0])
    Fb = np.diag(mf.mo_energy[1])
    # One-electron integrals in AO basis.
    h = mf.get_hcore()
    # One-electron part of the Moller-Plesset perturbation in MO basis: alpha and beta
    V1a = mf.mo_coeff[0].T @ h @ mf.mo_coeff[0] - Fa
    V1b = mf.mo_coeff[1].T @ h @ mf.mo_coeff[1] - Fb

    nmo = mf.mo_coeff[0].shape[1]
    # Two-electron parts of the Moller-Plesset perturbation in MO basis.
    V2aa = 0.5 * ao2mo.full(mol, mf.mo_coeff[0], aosym=1).reshape((nmo, nmo, nmo, nmo))
    V2ab = 0.5 * ao2mo.general(
        mol, (mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[1], mf.mo_coeff[1]), aosym=1
    ).reshape((nmo, nmo, nmo, nmo))
    V2bb = 0.5 * ao2mo.full(mol, mf.mo_coeff[1], aosym=1).reshape((nmo, nmo, nmo, nmo))

    # Calculate 1-RDM
    rdm1a, rdm1b = mp2density_conventional.make_rdm1_ump2(OH_radical_UMP2, mf.mo_energy)
    # Calculate 2-RDM
    rdm2aa, rdm2ab, rdm2bb = mp2density_conventional.make_rdm2_ump2(OH_radical_UMP2, mf.mo_energy)

    # 0th-order reference energy (calculated using PySCF): sum of occupied orbital energies.
    E0 = -46.754562761384
    # 1st-order reference energy (calculated using PySCF)
    E1 = -32.934885457887
    # 2nd-order reference energy (calculated using PySCF): MP2 correlation energy
    E2 = -0.1506146343736
    # 3rd-order reference energy (calculated using NWChem with MP3 in the TCE module): MP3 - MP2
    E3 = -0.0122538302248

    # Sanity check: 0th order + 1st order + nuclear repulsion energies must equal total HF energy
    assert isclose(E0 + E1 + mol.energy_nuc(), mf.e_tot, abs_tol=1e-8, rel_tol=0)
    # Sanity check: calculated MP2 energy equals reference number
    assert isclose(E2, OH_radical_UMP2.e_corr, abs_tol=1e-8, rel_tol=0)

    # Contract 1-RDM with one-particle part of the Moller-Plesset perturbation
    E = np.einsum("pq,pq", rdm1a, V1a)
    E += np.einsum("pq,pq", rdm1b, V1b)
    # Contract 2-RDM with two-particle part of the Moller-Plesset perturbation
    E += np.einsum("pqrs,pqrs", V2aa, rdm2aa)
    E += 2 * np.einsum("pqrs,pqrs", V2ab, rdm2ab)
    E += np.einsum("pqrs,pqrs", V2bb, rdm2bb)

    # E calculated above must equal 1 x 1st-order + 2 x 2nd order + 3 x 3rd order energies
    Eref = E1 + 2 * E2 + 3 * E3
    assert isclose(E, Eref, abs_tol=1e-7, rel_tol=0)
