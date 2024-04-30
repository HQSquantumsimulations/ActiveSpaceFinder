# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import gc
from math import isclose
from random import randrange

import numpy as np
import pytest
import scipy
from pyscf.gto import Mole
from pyscf.lib import current_memory
from pyscf.mp.dfump2_native import DFUMP2
from pyscf.scf import UHF
from scipy.stats import ortho_group

from asf import mp2density_conventional, pairinfo, rimp2_pairinfo
from asf.natorbs import unrestricted_natural_orbitals

from .fixtures.molecules import create_mol


def H2_UHF(distance=0.74, basis="def2-SVP"):
    """
    Perform UHF calculation for H2 molecule arbitrary distance.
    """
    mol = Mole()
    mol.atom = [("H", (0.0, 0.0, distance / 2)), ("H", (0.0, 0.0, -distance / 2))]
    mol.spin = 0
    mol.basis = basis
    mol.build()

    # Run UHF with tight threshold.
    mf = UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    # At long distances, H2 spin symmetry is broken. But UHF may converge to saddle point.
    # -> Perform stability analysis and reconverge MOs if necessary.
    mo_new = mf.stability()[0]
    if mo_new is not mf.mo_coeff:
        mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))

    return mf


def H2O_UHF(basis="def2-SVP"):
    """
    Perform UHF calculation for H2O at equilibrium distance.
    """
    mol = create_mol("water_2", basis=basis)
    # Run UHF with tight threshold.
    mf = UHF(mol).run(conv_tol=1e-12)
    return mf


def random_noncanonical(mf):
    """
    Generate random non-canonical spin-orbitals that do not mix occupied and virtual orbitals.
    """
    nmo = np.array(mf.mo_coeff).shape[2]
    nocc = np.count_nonzero(mf.mo_occ, axis=1)
    nvirt = nmo - nocc
    U = np.zeros((2, nmo, nmo))
    for s in 0, 1:
        U[s, : nocc[s], : nocc[s]] = ortho_group.rvs(nocc[s])
        U[s, nocc[s] :, nocc[s] :] = ortho_group.rvs(nvirt[s])
    mo_rotated = np.einsum("sxp,spq->sxq", mf.mo_coeff, U)
    return mo_rotated


def test_overlap_blocks(OH_radical_UHF):
    """
    Checking the computation of block-wise rotation matrices.
    """
    mf = OH_radical_UHF
    pt = DFUMP2(mf).run()
    nocca, noccb = pt.nocc
    for _ in range(10):
        mo_coeff = random_noncanonical(mf)
        (Uoa, Uob), (Uva, Uvb) = rimp2_pairinfo.overlap_blocks(pt, mo_coeff)
        rotated_occa = np.dot(pt.mo_coeff[0, :, :nocca], Uoa)
        assert np.allclose(rotated_occa, mo_coeff[0, :, :nocca], atol=1e-12, rtol=0)
        rotated_occb = np.dot(pt.mo_coeff[1, :, :noccb], Uob)
        assert np.allclose(rotated_occb, mo_coeff[1, :, :noccb], atol=1e-12, rtol=0)
        rotated_virta = np.dot(pt.mo_coeff[0, :, nocca:], Uva)
        assert np.allclose(rotated_virta, mo_coeff[0, :, nocca:], atol=1e-12, rtol=0)
        rotated_virtb = np.dot(pt.mo_coeff[1, :, noccb:], Uvb)
        assert np.allclose(rotated_virtb, mo_coeff[1, :, noccb:], atol=1e-12, rtol=0)
        with pytest.raises(ValueError):
            mo_coeff_mix = np.einsum("sxp,spq->sxq", mf.mo_coeff, ortho_group.rvs(pt.nmo, 2))
            rimp2_pairinfo.overlap_blocks(pt, mo_coeff_mix)


def test_make_rdm1_ump2(OH_radical_UMP2):
    """
    Checking the modified MP2 1-RDM with UHF reference.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object

    # Calculate the 1-RDM using RI-MP2. Use context manager for safety.
    with DFUMP2(mf) as pt_df:
        rdm1 = rimp2_pairinfo.make_rdm1_ump2(pt_df)

    # For reference: unrelaxed 1-RDM using RI-MP2. Context manager for safety.
    with DFUMP2(mf) as pt_df:
        rdm1_ur = pt_df.make_rdm1_unrelaxed()

    # Calculate the 1-RDM using conventional MP2.
    rdm1_conv = mp2density_conventional.make_rdm1_ump2(OH_radical_UMP2, mf.mo_energy)

    # Also the unrelaxed density with conventonal MP2 for reference.
    rdm1_ur_conv = OH_radical_UMP2.make_rdm1()

    # Check that the unrelaxed (occ.-occ., virt.-virt.) blocks match the RI-MP2 result.
    assert np.allclose(rdm1[0][:5, :5], rdm1_ur[0][:5, :5], atol=1e-12, rtol=0)
    assert np.allclose(rdm1[0][5:, 5:], rdm1_ur[0][5:, 5:], atol=1e-12, rtol=0)
    assert np.allclose(rdm1[1][:4, :4], rdm1_ur[1][:4, :4], atol=1e-12, rtol=0)
    assert np.allclose(rdm1[1][4:, 4:], rdm1_ur[1][4:, 4:], atol=1e-12, rtol=0)

    # Density matrix must be symmetric.
    assert np.allclose(rdm1[0], rdm1[0].T, atol=1e-14, rtol=0)
    assert np.allclose(rdm1[1], rdm1[1].T, atol=1e-14, rtol=0)
    # Trace of the density matrix must equal the number of electrons (5 spin-up, 4 spin-down).
    # Is most likely fulfilled already, as we checked against the unrelaxed density...
    assert isclose(np.trace(rdm1[0]), 5, abs_tol=1e-12, rel_tol=0)
    assert isclose(np.trace(rdm1[1]), 4, abs_tol=1e-12, rel_tol=0)

    # Now, we want to check the RDMs of RI-MP2 and conventional MP2 against each other.
    # There cannot be a perfect agreement due to the RI approximation, so we need a reasonable
    # threshold.
    # (1) Obtain only the additional (occ.-virt.) contributions beyond the unrelaxed density.
    Dvo = np.array(rdm1) - np.array(rdm1_ur)
    Dvo_conv = np.array(rdm1_conv) - np.array(rdm1_ur_conv)
    # (2) Check the norm of the conventionally computed one.
    assert isclose(np.linalg.norm(Dvo_conv), 0.02139396, abs_tol=1e-4, rel_tol=0)
    # (3) Total norm around 2e-2, so we use 5e-4 as an acceptable threshold for the difference
    #     between the respective RI-MP2 and conventional MP2 density contributions.
    assert isclose(np.linalg.norm(Dvo - Dvo_conv), 0.0, abs_tol=5e-4, rel_tol=0)

    # Checking a few natural occupation numbers (eigenvalues) calculated with this implementation.
    S = mf.get_ovlp()
    natocc, _ = unrestricted_natural_orbitals(rdm1, mf.mo_coeff, S)
    assert isclose(natocc[3], 1.97214745, abs_tol=1e-6, rel_tol=0)
    assert isclose(natocc[4], 0.99367654, abs_tol=1e-6, rel_tol=0)
    assert isclose(natocc[5], 0.02106853, abs_tol=1e-6, rel_tol=0)


def test_make_rdm2diag_0th_ump2(OH_radical_DFUMP2):
    """
    SCF contribution to the diagonal elements of the 2-RDM.
    """
    mf = OH_radical_DFUMP2._scf  # no public getter for SCF object

    nmo = mf.mo_coeff[0].shape[1]
    nocca, noccb = mf.mol.nelec

    rdm2s_ref = np.zeros((3, nmo, nmo, nmo, nmo))
    mp2density_conventional.make_rdm2_0th_ump2(nocca, noccb, rdm2s_ref)

    for _ in range(10):
        mo_coeff = random_noncanonical(mf)

        rdm2s_tf = pairinfo.transform_unrestricted_rdm2s(mf.mo_coeff, mo_coeff, mf.mol, rdm2s_ref)
        rdm2_diag_ref = np.einsum("sppqq->spq", rdm2s_tf)

        rdm2_diag = rimp2_pairinfo.make_rdm2diag_0th_ump2(OH_radical_DFUMP2, mo_coeff)

        assert np.allclose(rdm2_diag, rdm2_diag_ref, atol=1e-12, rtol=0.0)


def test_make_rdm2diag_separable_ump2(OH_radical_UHF):
    """
    Separable contributions to diagonal UHF-RI-MP(2, 1) 2-RDM vs. conventional code.
    """
    mf = OH_radical_UHF
    mo_scf = mf.mo_coeff

    pt = DFUMP2(mf).run()
    rdm1s_mp2 = rimp2_pairinfo.make_rdm1_ump2(pt)

    nmo = mo_scf[0].shape[1]
    nocca, noccb = mf.mol.nelec
    rdm2s_conv = np.zeros((3, nmo, nmo, nmo, nmo))
    mp2density_conventional.make_rdm2_2nd_separable_ump2(nocca, noccb, rdm1s_mp2, rdm2s_conv)

    for _ in range(10):
        U = np.array([ortho_group.rvs(nmo) for _ in (0, 1)])
        mo_coeff = np.einsum("sxp,spq->sxq", mo_scf, U)
        rdm2diag = rimp2_pairinfo.make_rdm2diag_separable_ump2(pt, rdm1s_mp2, mo_coeff)
        rdm2s_ref = pairinfo.transform_unrestricted_rdm2s(
            mf.mo_coeff, mo_coeff, mf.mol, rdm2s_conv
        )
        assert np.allclose(np.einsum("sppqq->spq", rdm2s_ref), rdm2diag, atol=1e-12, rtol=0.0)


def test_make_rdm2diag_vvvv_ump2(OH_radical_UMP2, OH_radical_DFUMP2):
    """
    Test virtual contributions to diagonal UHF-RI-MP(2, 1) 2-RDM vs. conventional code.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object

    nmo = OH_radical_UMP2.nmo[0]
    nocca, noccb = OH_radical_UMP2.nocc
    t2aa, t2ab, t2bb = OH_radical_UMP2.t2
    rdm2s_conv = np.zeros((3, nmo, nmo, nmo, nmo))
    mp2density_conventional.make_rdm2_tprod_ump2(nocca, noccb, t2aa, t2ab, t2bb, rdm2s_conv)

    for _ in range(10):
        mo_coeff = random_noncanonical(mf)

        rdm2diag = rimp2_pairinfo.make_rdm2diag_vvvv_ump2(OH_radical_DFUMP2, mo_coeff)

        rdm2s_ref = np.zeros_like(rdm2s_conv)
        rdm2s_ref[0, nocca:, nocca:, nocca:, nocca:] = rdm2s_conv[
            0, nocca:, nocca:, nocca:, nocca:
        ]
        rdm2s_ref[1, nocca:, nocca:, noccb:, noccb:] = rdm2s_conv[
            1, nocca:, nocca:, noccb:, noccb:
        ]
        rdm2s_ref[2, noccb:, noccb:, noccb:, noccb:] = rdm2s_conv[
            2, noccb:, noccb:, noccb:, noccb:
        ]

        rdm2s_tf = pairinfo.transform_unrestricted_rdm2s(mf.mo_coeff, mo_coeff, mf.mol, rdm2s_ref)
        assert np.allclose(np.einsum("sppqq->spq", rdm2s_tf), rdm2diag, atol=1e-8, rtol=1e-2)


def test_make_rdm2diag_oooo_ump2(OH_radical_UMP2, OH_radical_DFUMP2):
    """
    Test occupied contributions to diagonal UHF-RI-MP(2, 1) 2-RDM vs. conventional code.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object

    nmo = OH_radical_UMP2.nmo[0]
    nocca, noccb = OH_radical_UMP2.nocc
    t2aa, t2ab, t2bb = OH_radical_UMP2.t2
    rdm2s_conv = np.zeros((3, nmo, nmo, nmo, nmo))
    mp2density_conventional.make_rdm2_tprod_ump2(nocca, noccb, t2aa, t2ab, t2bb, rdm2s_conv)

    for _ in range(10):
        mo_coeff = random_noncanonical(mf)

        rdm2diag = rimp2_pairinfo.make_rdm2diag_oooo_ump2(OH_radical_DFUMP2, mo_coeff)

        rdm2s_ref = np.zeros_like(rdm2s_conv)
        rdm2s_ref[0, :nocca, :nocca, :nocca, :nocca] = rdm2s_conv[
            0, :nocca, :nocca, :nocca, :nocca
        ]
        rdm2s_ref[1, :nocca, :nocca, :noccb, :noccb] = rdm2s_conv[
            1, :nocca, :nocca, :noccb, :noccb
        ]
        rdm2s_ref[2, :noccb, :noccb, :noccb, :noccb] = rdm2s_conv[
            2, :noccb, :noccb, :noccb, :noccb
        ]

        rdm2s_tf = pairinfo.transform_unrestricted_rdm2s(mf.mo_coeff, mo_coeff, mf.mol, rdm2s_ref)
        assert np.allclose(np.einsum("sppqq->spq", rdm2s_tf), rdm2diag, atol=1e-8, rtol=1e-2)


def test_make_rdm2diag_oooo_ump2_memory(OH_radical_UHF):
    """
    Test the memory buffering algorithm in the function make_rdm2diag_oooo_ump2.
    """
    # Set up the reference result, which is tested elsewhere.
    mf = OH_radical_UHF
    pt = DFUMP2(mf).run()
    ref_val = rimp2_pairinfo.make_rdm2diag_oooo_ump2(pt, mf.mo_coeff)

    # Minimum required memory for the buffer is going to be 16 * nocc * naux.
    # Take half the value, in MB, as a memory increment.
    memory_increment = 8e-6 * pt.nocc[1] * pt.auxmol.nao

    # Hack the upper memory limit in the DFUMP2 object to equal the amount of currently used memory.
    gc.collect()
    pt.max_memory = current_memory()[0]
    # We have no memory left for buffering, so this must result in an error.
    with pytest.raises(MemoryError):
        rimp2_pairinfo.make_rdm2diag_oooo_ump2(pt, mf.mo_coeff)

    # Increase the memory by the previously defined increment. This will also not be enough.
    gc.collect()
    pt.max_memory = current_memory()[0] + memory_increment
    with pytest.raises(MemoryError):
        rimp2_pairinfo.make_rdm2diag_oooo_ump2(pt, mf.mo_coeff)

    # Increase the upper memory limit until the function can just about run without failing.
    for _ in range(100):
        pt.max_memory += memory_increment
        gc.collect()
        try:
            val = rimp2_pairinfo.make_rdm2diag_oooo_ump2(pt, mf.mo_coeff)
        except MemoryError:
            pass
        else:
            # Check that the result matches the reference value computed previously.
            assert np.allclose(val, ref_val, atol=1e-12, rtol=0.0)
            break

    # Raise the memory limit in increasingly large steps, and verify the correctness of the result.
    for n in range(10):
        pt.max_memory += memory_increment * 1.5**n
        gc.collect()
        val = rimp2_pairinfo.make_rdm2diag_oooo_ump2(pt, mf.mo_coeff)
        assert np.allclose(val, ref_val, atol=1e-12, rtol=0.0)


def test_make_rdm2diag_oovv_ump2(OH_radical_UMP2, OH_radical_DFUMP2):
    """
    Test occupied contributions to diagonal UHF-RI-MP(2, 1) 2-RDM vs. conventional code.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object

    nmo = OH_radical_UMP2.nmo[0]
    nocca, noccb = OH_radical_UMP2.nocc
    t2aa, t2ab, t2bb = OH_radical_UMP2.t2
    rdm2s_conv = np.zeros((3, nmo, nmo, nmo, nmo))
    mp2density_conventional.make_rdm2_tprod_ump2(nocca, noccb, t2aa, t2ab, t2bb, rdm2s_conv)

    for _ in range(10):
        mo_coeff = random_noncanonical(mf)

        rdm2diag = rimp2_pairinfo.make_rdm2diag_oovv_ump2(OH_radical_DFUMP2, mo_coeff)

        rdm2s_ref = np.zeros_like(rdm2s_conv)
        rdm2s_ref[0, :nocca, :nocca, nocca:, nocca:] = rdm2s_conv[
            0, :nocca, :nocca, nocca:, nocca:
        ]
        rdm2s_ref[0, nocca:, nocca:, :nocca, :nocca] = rdm2s_conv[
            0, nocca:, nocca:, :nocca, :nocca
        ]
        rdm2s_ref[0, :nocca, nocca:, nocca:, :nocca] = rdm2s_conv[
            0, :nocca, nocca:, nocca:, :nocca
        ]
        rdm2s_ref[0, nocca:, :nocca, :nocca, nocca:] = rdm2s_conv[
            0, nocca:, :nocca, :nocca, nocca:
        ]
        rdm2s_ref[1, :nocca, :nocca, noccb:, noccb:] = rdm2s_conv[
            1, :nocca, :nocca, noccb:, noccb:
        ]
        rdm2s_ref[1, nocca:, nocca:, :noccb, :noccb] = rdm2s_conv[
            1, nocca:, nocca:, :noccb, :noccb
        ]
        rdm2s_ref[1, :nocca, nocca:, noccb:, :noccb] = rdm2s_conv[
            1, :nocca, nocca:, noccb:, :noccb
        ]
        rdm2s_ref[1, nocca:, :nocca, :noccb, noccb:] = rdm2s_conv[
            1, nocca:, :nocca, :noccb, noccb:
        ]
        rdm2s_ref[2, :noccb, :noccb, noccb:, noccb:] = rdm2s_conv[
            2, :noccb, :noccb, noccb:, noccb:
        ]
        rdm2s_ref[2, noccb:, noccb:, :noccb, :noccb] = rdm2s_conv[
            2, noccb:, noccb:, :noccb, :noccb
        ]
        rdm2s_ref[2, :noccb, noccb:, noccb:, :noccb] = rdm2s_conv[
            2, :noccb, noccb:, noccb:, :noccb
        ]
        rdm2s_ref[2, noccb:, :noccb, :noccb, noccb:] = rdm2s_conv[
            2, noccb:, :noccb, :noccb, noccb:
        ]

        rdm2s_tf = pairinfo.transform_unrestricted_rdm2s(mf.mo_coeff, mo_coeff, mf.mol, rdm2s_ref)
        for s in 0, 1, 2:
            assert np.allclose(
                np.einsum("ppqq->pq", rdm2s_tf[s]), rdm2diag[s], atol=1e-8, rtol=1e-2
            )


def test_make_rdm2diag_oovv_ump2_memory(OH_radical_UHF):
    """
    Test the memory buffering algorithm in the function make_rdm2diag_oovv_ump2.
    """
    # Set up the reference result, which is tested elsewhere.
    mf = OH_radical_UHF
    pt = DFUMP2(mf).run()
    ref_val = rimp2_pairinfo.make_rdm2diag_oovv_ump2(pt, mf.mo_coeff)

    # Minimum required memory for the buffer is going to be 4 * nocc * nvirt.
    # Take half the value, in MB, as a memory increment.
    memory_increment = 4e-6 * pt.nocc[1] * (pt.nmo - pt.nocc[0])

    # Hack the upper memory limit in the DFUMP2 object to equal the amount of currently used memory.
    gc.collect()
    pt.max_memory = current_memory()[0]
    # We have no memory left for buffering, so this must result in an error.
    with pytest.raises(MemoryError):
        rimp2_pairinfo.make_rdm2diag_oovv_ump2(pt, mf.mo_coeff)

    # Increase the memory by the previously defined increment. This will also not be enough.
    gc.collect()
    pt.max_memory = current_memory()[0] + memory_increment
    with pytest.raises(MemoryError):
        rimp2_pairinfo.make_rdm2diag_oovv_ump2(pt, mf.mo_coeff)

    # Increase the upper memory limit until the function can just about run without failing.
    for _ in range(100):
        pt.max_memory += memory_increment
        gc.collect()
        try:
            val = rimp2_pairinfo.make_rdm2diag_oovv_ump2(pt, mf.mo_coeff)
        except MemoryError:
            pass
        else:
            # Check that the result matches the reference value computed previously.
            assert np.allclose(val, ref_val, atol=1e-12, rtol=0.0)
            break

    # Raise the memory limit in increasingly large steps, and verify the correctness of the result.
    for n in range(10):
        pt.max_memory += memory_increment * 1.5**n
        gc.collect()
        val = rimp2_pairinfo.make_rdm2diag_oovv_ump2(pt, mf.mo_coeff)
        assert np.allclose(val, ref_val, atol=1e-12, rtol=0.0)


def test_diag_cumulant_ump2(OH_radical_UMP2, OH_radical_DFUMP2):
    """
    Comparing the cumulant from UHF-RI-MP2(2, 1) with it non-RI counterpart.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object

    # Calculate the full UHF-MP(2, 1) cumulant with the conventional (non-RI) algorithm.
    rdm1_conv = mp2density_conventional.make_rdm1_ump2(OH_radical_UMP2, mf.mo_energy)
    rdm2_conv = mp2density_conventional.make_rdm2_ump2(OH_radical_UMP2, mf.mo_energy)
    cu_conv = rdm2_conv.copy()
    cu_conv[0] = cu_conv[0] - np.einsum("pq,rs->pqrs", rdm1_conv[0], rdm1_conv[0])
    cu_conv[0] = cu_conv[0] + np.einsum("ps,rq->pqrs", rdm1_conv[0], rdm1_conv[0])
    cu_conv[1] = cu_conv[1] - np.einsum("pq,rs->pqrs", rdm1_conv[0], rdm1_conv[1])
    cu_conv[2] = cu_conv[2] - np.einsum("pq,rs->pqrs", rdm1_conv[1], rdm1_conv[1])
    cu_conv[2] = cu_conv[2] + np.einsum("ps,rq->pqrs", rdm1_conv[1], rdm1_conv[1])

    # Use a deliberately large auxiliary set to reduce deviations between MP2 and RI-MP2.
    for _ in range(10):
        mo_coeff = random_noncanonical(mf)

        # Diagonal elements of the cumulant with RI-MP(2, 1) in the transformed MO basis.
        cudiag = rimp2_pairinfo.diag_cumulant_ump2(OH_radical_DFUMP2, mo_coeff)

        # Diagonal elements of the reference cumulant in the transformed basis.
        cu_ref = pairinfo.transform_unrestricted_rdm2s(mf.mo_coeff, mo_coeff, mf.mol, cu_conv)
        cudiag_ref = np.einsum("sppqq->spq", cu_ref)

        # Allow for some discrepancy due to the differences between the RI and non-RI methods.
        assert np.allclose(cudiag, cudiag_ref, atol=1e-8, rtol=1e-2)


def test_UHF_corresponding_orbitals():
    """
    Testing the construction of unrestricted corresponding orbitals.
    """
    # Total dimensions of MO space for testing ranging from 1 to 10.
    for N in range(1, 11):

        # Construct symmetric, positive definite matrix to represent an overlap matrix.
        # Small risk of ending up with a near-singular matrix.
        R = np.random.random((N, N))
        S = R @ R.T

        # Inverse square root of S gives a set of vectors that are orthonormal with respect to S.
        # S^-1/2 @ S @ S^-1/2 == 1
        orth_coeff = scipy.linalg.inv(scipy.linalg.sqrtm(S))

        # Randomly rotated set of orthonormal "orbitals", different for alpha and beta.
        # ortho_group.rvs requires N >= 2.
        if N > 1:
            mo_a = orth_coeff @ ortho_group.rvs(N)
            mo_b = orth_coeff @ ortho_group.rvs(N)
            mo_coeff = np.array((mo_a, mo_b))
        else:
            mo_coeff = np.array((orth_coeff,) * 2)

        # Iterate over sensible combinations of numbers of occupied alpha and beta orbitals.
        for nocca in range(1, N):
            for noccb in range(0, nocca + 1):
                nocc = np.array((nocca, noccb))

                # Number of virtual orbitals.
                nvirt = N - nocc
                nvirta, nvirtb = nvirt

                # Calculate the actual corresponding orbitals and their associated singular values.
                sigma, C = rimp2_pairinfo.UHF_corresponding_orbitals(mo_coeff, nocc, S)

                assert sigma.shape == (N,)
                assert C.shape == (2, N, N)

                # Testing for orthonormality of the MO coefficients.
                assert np.allclose(C[0].T @ S @ C[0], np.eye(N), atol=1e-8, rtol=0)
                assert np.allclose(C[1].T @ S @ C[1], np.eye(N), atol=1e-8, rtol=0)

                # Overlap of occupied alpha and beta orbitals is a rectangular matrix with the
                # singular values on its diagonal (at the top) and zeros everywhere else.
                ref_Sab_occ = np.zeros((nocca, noccb))
                ref_Sab_occ[np.arange(noccb), np.arange(noccb)] = sigma[:noccb]
                Sab_occ = C[0, :, :nocca].T @ S @ C[1, :, :noccb]
                assert np.allclose(Sab_occ, ref_Sab_occ, atol=1e-8, rtol=0)

                # Overlap of virtual alpha and beta orbitals is a rectangular matrix with the
                # singular values on its diagonal (to the right) and zeros everywhere else.
                ref_Sab_virt = np.zeros(nvirt)
                ref_Sab_virt[np.arange(nvirta), np.arange(nvirtb - nvirta, nvirtb)] = sigma[nocca:]
                Sab_virt = C[0, :, nocca:].T @ S @ C[1, :, noccb:]
                assert np.allclose(Sab_virt, ref_Sab_virt, atol=1e-8, rtol=0)

                # Occupied singular values in descending order.
                if noccb >= 2:
                    assert np.all(sigma[0 : noccb - 1] >= sigma[1:noccb])

                # Zero singular values for each unpaired electron.
                if nocca - noccb >= 1:
                    assert np.all(sigma[noccb:nocca] == 0.0)

                # Virtual singular values in ascending order.
                if N - nocca >= 2:
                    assert np.all(sigma[nocca : N - 1] <= sigma[nocca + 1 : N])


@pytest.mark.xfail(strict=False, reason="different stable SCF solution for stretched H2")
def test_corresponding_orbital_subspaces(OH_radical_UHF, allyl_UHF):
    """
    Testing the function corresponding_orbital_subspaces.
    """

    def square_diagonal(A):
        """
        Check that a square matrix is diagonal (to within a threshold).
        """
        assert A.ndim == 2
        rows, cols = A.shape
        if rows != cols:
            return False
        return np.allclose(np.diag(np.diag(A)), A, atol=1e-8, rtol=0)

    def tall_diagonal(A, top=True):
        """
        Check that a tall rectangular matrix is diagonal.

        If top is True, the non-zero diagonal is at the top, otherwise it is at the bottom.
        """
        assert A.ndim == 2
        rows, cols = A.shape
        if cols > rows:
            return False
        if top:
            top_is_diagonal = square_diagonal(A[:cols, :])
            bottom_is_zero = np.allclose(A[cols:, :], 0.0, atol=1e-8, rtol=0)
            return top_is_diagonal and bottom_is_zero
        else:
            return tall_diagonal(np.flip(A, axis=(0, 1)))

    def verify_results(pt, ref_active):
        """
        The main "checking" function".

        Takes a DFUMP2 instance and a list of orbitals in the refence minimum active space.
        """
        # Unrelaxed MP2 density matrix in AO basis.
        rdm1ao = pt.make_rdm1_unrelaxed(ao_repr=True)

        # Overlap matrix of atomic orbitals.
        S = pt.mol.intor_symmetric("int1e_ovlp")

        # Numbers of occupied alpha and beta orbitals.
        nocca, noccb = pt.nocc
        # Total number of orbitals.
        nmo = pt.nmo

        # Iterate over combinations of active and inactive orbitals diagonalizing or not
        # diagonalizing the MP2 1-RDM.
        for inactive_mp2no in False, True:
            for active_mp2no in False, True:

                # Calculate the actual data to be tested:
                # active: list of spin orbitals (both alpha and beta) in the minimal active space
                # mo_coeff: MO coefficients containing the transformed corresponding spin orbitals
                # mo_re: Restricted orbitals that the spin orbitals can be mapped onto.
                active, mo_coeff, mo_re = rimp2_pairinfo.corresponding_orbital_subspaces(
                    pt, active_mp2no=active_mp2no, inactive_mp2no=inactive_mp2no
                )
                # Indices of "inactive" MOs.
                inactive = np.setdiff1d(np.arange(pt.nmo), active)

                # Checking if the correct orbital indices have been chosen for the active space.
                assert np.array_equal(active, ref_active)

                # Overlap between spatial parts of alpha and beta orbitals in mo_coeff.
                Sab = mo_coeff[0].T @ S @ mo_coeff[1]

                # Indices of active occupied orbitals
                active_occupied_a = np.intersect1d(active, np.arange(nocca))
                active_occupied_b = np.intersect1d(active, np.arange(noccb))

                # Indices of active virtual orbitals.
                active_virtual_a = np.intersect1d(active, np.arange(nocca, nmo))
                active_virtual_b = np.intersect1d(active, np.arange(noccb, nmo))

                # Indices of inactive occupied orbitals.
                inactive_occupied_a = np.intersect1d(inactive, np.arange(nocca))
                inactive_occupied_b = np.intersect1d(inactive, np.arange(noccb))

                # Indices of inactive virtual orbitals.
                inactive_virtual_a = np.intersect1d(inactive, np.arange(nocca, nmo))
                inactive_virtual_b = np.intersect1d(inactive, np.arange(noccb, nmo))

                if active_mp2no:
                    # Iterate over alpha and beta orbitals.
                    for s in (0, 1):
                        # MP2 1-RDM in the subspace of active spin orbitals must be diagonal.
                        mos_act = mo_coeff[s][:, active]
                        rdm1_act = mos_act.T @ S @ rdm1ao[s] @ S @ mos_act
                        assert square_diagonal(rdm1_act)
                else:
                    # Overlap matrix of spatial parts of alpha and beta orbitals is a tall
                    # rectangular matrix with diagonal at the top for active occupied orbitals...
                    assert tall_diagonal(Sab[active_occupied_a, :][:, active_occupied_b])
                    # ... and wide matrix with diagonal on the right for active virtual MOs.
                    assert tall_diagonal(
                        Sab[active_virtual_a, :][:, active_virtual_b].T, top=False
                    )

                if inactive_mp2no:
                    # Spin-free MP2 1-RDM transformed to mixed basis, consisting of the spatial part of
                    # the alpha orbitals (rows) and the spatial part of the beta orbitals (columns).
                    rdm1ab = mo_coeff[0].T @ S @ (rdm1ao[0] + rdm1ao[1]) @ S @ mo_coeff[1]

                    # Spin-free MP2 1-RDM projected to spatial alpha / beta orbitals is a diagonal
                    # square matrix in occupied and virtual inactive subspaces.
                    assert square_diagonal(rdm1ab[inactive_occupied_a, :][:, inactive_occupied_b])
                    assert square_diagonal(rdm1ab[inactive_virtual_a, :][:, inactive_virtual_b])
                else:
                    # Overlap matrix of spatial alpha and beta orbitals is a diagonal square matrix
                    # in the occupied and virtual inactive subspaces.
                    assert square_diagonal(Sab[inactive_occupied_a, :][:, inactive_occupied_b])
                    assert square_diagonal(Sab[inactive_virtual_a, :][:, inactive_virtual_b])

                # Restricted MOs are orthonormal.
                assert np.allclose(mo_re.T @ S @ mo_re - np.eye(nmo), 0.0, atol=1e-12, rtol=0.0)
                # For inactive orbitals, restricted and unrestricted orbitals with largest
                # (absolute) overlap have the same column indices.
                S_re_a = mo_re[:, inactive].T @ S @ mo_coeff[0][:, inactive]
                assert np.array_equal(np.argmax(S_re_a, axis=0), np.arange(len(inactive)))

    # Test calculation for H2 at equilibrium geometry with STO-3G basis.
    with DFUMP2(H2_UHF(basis="STO-3G"), auxbasis="def2-SVP-RI") as pt:
        # Empty active space.
        verify_results(pt, [])

    # Test calculation for H2 at 10A separation with def2-SVP basis.
    with DFUMP2(H2_UHF(distance=10.0)) as pt:
        # Active space with bonding and antibonding orbital.
        verify_results(pt, [0, 1])

    # Test calculation for H2O at equilibrium geometry with STO-3G basis.
    with DFUMP2(H2O_UHF(basis="6-31G"), auxbasis="def2-SVP-RI") as pt:
        # Empty active space.
        verify_results(pt, [])

    # Test calculation for OH radical with def2-SVP basis.
    with DFUMP2(OH_radical_UHF) as pt:
        # Active space containing the unpaired electron in its orbital only.
        verify_results(pt, [4])

    # Test calculation for allyl radical with def2-SVP basis.
    with DFUMP2(allyl_UHF) as pt:
        # Active space containing the three pi orbitals.
        verify_results(pt, [10, 11, 12])


@pytest.mark.parametrize(
    "mol",
    [
        create_mol("formaldehyde_2"),
        create_mol("cyanide"),
        create_mol("benzyl_radical"),
        create_mol("oxygen"),
    ],
)
def test_ur_to_re_pairwise(mol):
    # Build the molecule object and run UHF calculation.
    mol.build()
    N = mol.nao
    mf = UHF(mol).run()

    # Construct MOs with 1-1 match via corresponding transformation of the UHF orbitals.
    S = mf.get_ovlp()
    _, mo_corr = rimp2_pairinfo.UHF_corresponding_orbitals(mf.mo_coeff, mf.nelec, S)

    # Restricted orbitals will be constructed from unrestricted orbitals and tested. Repeat this
    # many times and flip the overall signs of spin orbitals randomly.
    for _ in range(1000):

        # Construct restricted orbitals (to be tested) from the unrestricted orbitals.
        mo_orth = rimp2_pairinfo.ur_to_re_pairwise(mo_corr, S)

        # (1) Are the restricted output orbitals orthonormal?
        assert np.allclose(mo_orth.T @ S @ mo_orth, np.eye(N), atol=1e-8, rtol=0.0)

        # (2) Is each restricted MO with index i most closely matching with spin orbitals i,alpha
        # and i,beta?
        ovlp_alpha = np.abs(mo_orth.T @ S @ mo_corr[0])
        ovlp_beta = np.abs(mo_orth.T @ S @ mo_corr[1])
        assert np.array_equal(np.argmax(ovlp_alpha, axis=0), np.arange(N))
        assert np.array_equal(np.argmax(ovlp_alpha, axis=1), np.arange(N))
        assert np.array_equal(np.argmax(ovlp_beta, axis=0), np.arange(N))
        assert np.array_equal(np.argmax(ovlp_beta, axis=1), np.arange(N))

        # One spin orbital is selected randomly and its overall sign flipped.
        spin = randrange(2)
        orbital = randrange(N)
        mo_corr[spin, :, orbital] *= -1.0
