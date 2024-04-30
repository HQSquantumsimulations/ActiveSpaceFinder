# Copyright © 2020-2022 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Implementation of modified Møller-Plesset reduced density matrices with conventional MP2."""

from typing import Union

import numpy as np
from pyscf import ao2mo
from pyscf.gto import Mole
from pyscf.mp.mp2 import MP2 as RMP2class
from pyscf.mp.ump2 import UMP2 as UMP2class


def make_rdm1_ump2(pt: UMP2class, mo_energy: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Calculate MP(2, 1) one-particle reduced density matrix (UHF reference).

    Args:
        pt:         Unrestricted MP2 object.
        mo_energy:  MO orbital energies in tuple of vectors (alpha MO energies, beta MO energies).

    Returns:
        1-RDM in MO basis as an array of dimensions 2 x N(MO) x N(MO), with alpha and beta parts.
    """
    # Molecular data.
    mol = pt.mol

    # MP2 amplitudes: t^ij_ab -> t[i, j, a, b] with ij occupied, ab virtual
    # t2aa: pure spin up, t2bb: pure spin-down, t2ab: mixed-spin
    # Indices of t2ab: t[occ. alpha, occ. beta, virt. alpha, virt. beta]
    t2aa, t2ab, t2bb = pt.t2

    # Numbers of occupied and virtual MOs.
    nocca, noccb, nvirta, nvirtb = t2ab.shape

    # MO coefficient matrices (occupied and virtual, alpha and beta)
    Coa = pt.mo_coeff[0][:, :nocca]
    Cob = pt.mo_coeff[1][:, :noccb]
    Cva = pt.mo_coeff[0][:, nocca:]
    Cvb = pt.mo_coeff[1][:, noccb:]

    # Unrelaxed MP2 1-RDM: spin-up and spin-down parts in UHF MO basis.
    # Only the occupied-occupied and virtual-virtual blocks are non-zero.
    rdm1a, rdm1b = pt.make_rdm1(ao_repr=False)

    # Calculate ERIs that we will need later.
    eri_ooov_aa = ao2mo.general(mol, (Coa, Coa, Coa, Cva), aosym=1).reshape(
        (nocca, nocca, nocca, nvirta)
    )
    eri_ovvv_aa = ao2mo.general(mol, (Coa, Cva, Cva, Cva), aosym=1).reshape(
        (nocca, nvirta, nvirta, nvirta)
    )
    eri_ooov_ab = ao2mo.general(mol, (Coa, Coa, Cob, Cvb), aosym=1).reshape(
        (nocca, nocca, noccb, nvirtb)
    )
    eri_ooov_ba = ao2mo.general(mol, (Cob, Cob, Coa, Cva), aosym=1).reshape(
        (noccb, noccb, nocca, nvirta)
    )
    eri_ovvv_ab = ao2mo.general(mol, (Coa, Cva, Cvb, Cvb), aosym=1).reshape(
        (nocca, nvirta, nvirtb, nvirtb)
    )
    eri_ovvv_ba = ao2mo.general(mol, (Cob, Cvb, Cva, Cva), aosym=1).reshape(
        (noccb, nvirtb, nvirta, nvirta)
    )
    eri_ooov_bb = ao2mo.general(mol, (Cob, Cob, Cob, Cvb), aosym=1).reshape(
        (noccb, noccb, noccb, nvirtb)
    )
    eri_ovvv_bb = ao2mo.general(mol, (Cob, Cvb, Cvb, Cvb), aosym=1).reshape(
        (noccb, nvirtb, nvirtb, nvirtb)
    )

    # Calculate the alpha part of the occupied-virtual block in the modified 1-RDM.
    numerator = np.einsum("iklc,klac->ia", eri_ooov_aa, t2aa)
    numerator -= np.einsum("kcad,kicd->ia", eri_ovvv_aa, t2aa)
    numerator += np.einsum("iklc,klac->ia", eri_ooov_ab, t2ab)
    numerator -= np.einsum("kcad,ikdc->ia", eri_ovvv_ba, t2ab)
    for i in range(nocca):
        for a in range(nvirta):
            rdm1a[i, nocca + a] = numerator[i, a] / (mo_energy[0][nocca + a] - mo_energy[0][i])
            rdm1a[nocca + a, i] = rdm1a[i, nocca + a]

    # Calculate the beta part of the occupied-virtual block in the modified 1-RDM.
    numerator = np.einsum("iklc,klac->ia", eri_ooov_bb, t2bb)
    numerator -= np.einsum("kcad,kicd->ia", eri_ovvv_bb, t2bb)
    numerator += np.einsum("iklc,lkca->ia", eri_ooov_ba, t2ab)
    numerator -= np.einsum("kcad,kicd->ia", eri_ovvv_ab, t2ab)
    for i in range(noccb):
        for a in range(nvirtb):
            rdm1b[i, noccb + a] = numerator[i, a] / (mo_energy[1][noccb + a] - mo_energy[1][i])
            rdm1b[noccb + a, i] = rdm1b[i, noccb + a]

    # Return the alpha and beta parts as a tuple of matrices.
    return np.array([rdm1a, rdm1b])


def make_rdm1_rmp2(pt: RMP2class, mo_energy: np.ndarray) -> np.ndarray:
    """Calculate MP(2, 1) one-particle reduced density matrix (RHF reference).

    Args:
        pt:         Restricted MP2 object.
        mo_energy:  MO orbital energies as a vector.

    Returns:
        1-RDM in MO basis
    """
    # Molecular data.
    mol = pt.mol

    # Spin-free MP2 amplitudes: t^ij_ab -> t[i, j, a, b] with ij occupied, ab virtual
    t2 = pt.t2

    # Numbers of occupied and virtual MOs.
    nocc = pt.get_nocc()
    nvirt = pt.get_nmo() - nocc

    # Occupied and virtual MO coefficient matrices
    Co = pt.mo_coeff[:, :nocc]
    Cv = pt.mo_coeff[:, nocc:]

    # Unrelaxed MP2 1-RDM in RHF MO basis.
    # Only the occupied-occupied and virtual-virtual blocks are non-zero.
    rdm1 = pt.make_rdm1(ao_repr=False)

    # Calculate ERIs for contraction (3 occ., 1 virt.) and (1 occ., 3. virt)
    eri_ooov = ao2mo.general(mol, (Co, Co, Co, Cv), aosym=1).reshape((nocc, nocc, nocc, nvirt))
    eri_ovvv = ao2mo.general(mol, (Co, Cv, Cv, Cv), aosym=1).reshape((nocc, nvirt, nvirt, nvirt))

    # Now calculate the density matrix contribution.
    t2c = 4 * t2 - 2 * t2.transpose((0, 1, 3, 2))
    numerator = np.einsum("iklc,klac->ia", eri_ooov, t2c)
    numerator -= np.einsum("kcad,kicd->ia", eri_ovvv, t2c)
    for i in range(nocc):
        for a in range(nvirt):
            rdm1[i, nocc + a] = numerator[i, a] / (mo_energy[nocc + a] - mo_energy[i])
            rdm1[nocc + a, i] = rdm1[i, nocc + a]

    return rdm1


def make_rdm2_ump2(pt: UMP2class, mo_energy: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Calculate MP(2, 1) two-particle reduced density matrix (UHF reference).

    Args:
        pt:         Unrestricted MP2 object.
        mo_energy:  MO orbital energies in tuple of vectors (alpha MO energies, beta MO energies).

    Returns:
        2-RDM in MO basis as an array of dimensions 3 x N(MO) x N(MO) x N(MO) x N(MO).
            rdm2s[0, p, q, r, s] = <p(alpha)+ r(alpha)+ s(alpha) q(alpha)>
            rdm2s[1, p, q, r, s] = <p(alpha)+ r(beta)+ s(beta) q(alpha)>
            rdm2s[2, p, q, r, s] = <p(beta)+ r(beta)+ s(beta) q(beta)>
        This follows the convention of PySCF for reduced density matrices.

    Raises:
        ValueError: Invalid input.
    """
    # MP2 amplitudes: t^ij_ab -> t[i, j, a, b] with ij occupied, ab virtual
    # t2aa: pure spin up, t2bb: pure spin-down, t2ab: mixed-spin
    # Indices of t2ab: t[occ. alpha, occ. beta, virt. alpha, virt. beta]
    t2aa, t2ab, t2bb = pt.t2

    # Get total number of MOs.
    nmo_alpha, nmo_beta = pt.get_nmo()
    if nmo_alpha != nmo_beta:
        raise ValueError("Different numbers of alpha and beta MOs.")
    nmo = nmo_alpha

    # Numbers of occupied alpha and beta MOs.
    nocca, noccb = pt.get_nocc()

    # Alpha and beta parts of the 1-RDM in unrestricted MO basis.
    rdm1s = make_rdm1_ump2(pt, mo_energy)

    # Initialize the tensor to store the 2-RDM.
    rdm2s = np.zeros((3, nmo, nmo, nmo, nmo))

    # Oth-order contribution to the 2-RDM (Hartree-Fock terms).
    make_rdm2_0th_ump2(nocca, noccb, rdm2s)

    # 1st-order contribution to the 2-RDM.
    make_rdm2_1st_ump2(t2aa, t2ab, t2bb, rdm2s)

    # Products of 2nd order 1-RDM with Kronecker deltas.
    make_rdm2_2nd_separable_ump2(nocca, noccb, rdm1s, rdm2s)

    # Tensor products of MP2 amplitudes.
    make_rdm2_tprod_ump2(nocca, noccb, t2aa, t2ab, t2bb, rdm2s)

    # Contractions of ERIs with MP2 amplitudes.
    make_rdm2_contractions_ump2(
        pt.mol, nocca, noccb, pt.mo_coeff, mo_energy, t2aa, t2ab, t2bb, rdm2s
    )

    return rdm2s


def make_rdm2_0th_ump2(nocca: int, noccb: int, rdm2s: np.ndarray) -> None:
    """Zeroth-order contribution to the UHF 2-RDM: Hartree-Fock terms.

    Args:
        nocca:  Number of occupied spin-up orbitals.
        noccb:  Number of occupied spin-down orbitals.
        rdm2s:  The 2-RDM contribution is added to this array.
    """
    # Pure alpha Hartree-Fock contribution.
    for i in range(nocca):
        for j in range(nocca):
            rdm2s[0, i, i, j, j] += 1.0
            rdm2s[0, i, j, j, i] -= 1.0

    # Mixed alpha-beta Hartree-Fock contribution.
    for i in range(nocca):
        for j in range(noccb):
            rdm2s[1, i, i, j, j] += 1.0

    # Pure beta Hartree-Fock contribution.
    for i in range(noccb):
        for j in range(noccb):
            rdm2s[2, i, i, j, j] += 1.0
            rdm2s[2, i, j, j, i] -= 1.0

    return


def make_rdm2_1st_ump2(
    t2aa: np.ndarray, t2ab: np.ndarray, t2bb: np.ndarray, rdm2s: np.ndarray
) -> None:
    """First-order contribution to the UHF-MP(2, 1) 2-RDM.

    Args:
        t2aa:   All-alpha MP2 amplitudes.
        t2ab:   Alpha-beta MP2 amplitudes.
        t2bb:   All-beta MP2 amplitudes.
        rdm2s:  The 2-RDM contribution is added to this array.
    """
    # Infer numbers of occupied orbitals from the shapes of the amplitude tensors.
    nocca = t2aa.shape[0]
    noccb = t2bb.shape[0]

    # Assign pure alpha parts of the 2-RDM.
    rdm2s[0, :nocca, nocca:, :nocca, nocca:] += t2aa.transpose((0, 2, 1, 3))
    rdm2s[0, nocca:, :nocca, nocca:, :nocca] += t2aa.transpose((2, 0, 3, 1))

    # Assign mixed alpha-beta parts of the 2-RDM.
    rdm2s[1, :nocca, nocca:, :noccb, noccb:] += t2ab.transpose((0, 2, 1, 3))
    rdm2s[1, nocca:, :nocca, noccb:, :noccb] += t2ab.transpose((2, 0, 3, 1))

    # Assign pure beta parts of the 2-RDM.
    rdm2s[2, :noccb, noccb:, :noccb, noccb:] += t2bb.transpose((0, 2, 1, 3))
    rdm2s[2, noccb:, :noccb, noccb:, :noccb] += t2bb.transpose((2, 0, 3, 1))

    return


def make_rdm2_2nd_separable_ump2(
    nocca: int, noccb: int, rdm1s_mp2: np.ndarray, rdm2s: np.ndarray
) -> None:
    """2nd order contributions to the UHF-MP(2, 1) 2-RDM that separate into products of 1-RDMs.

    Args:
        nocca:      Number of occupied alpha orbitals.
        noccb:      Number of occupied beta orbitals.
        rdm1s_mp2:  Full UHF-MP(2, 1) 1-RDM, dimensions 2 x N(MO) x N(MO).
        rdm2s:      The 2-RDM contribution is added to this array.
    """
    # We will need the 2nd order components of the 1-RDM only. For this reason, the 0th order (HF)
    # component is subtracted, consisting of delta_ij for occupied orbitals.
    rdm1s_2nd = rdm1s_mp2.copy()
    for i in range(nocca):
        rdm1s_2nd[0, i, i] -= 1.0
    for i in range(noccb):
        rdm1s_2nd[1, i, i] -= 1.0

    for i in range(nocca):
        # pure alpha contributions
        rdm2s[0, i, i, :, :] += rdm1s_2nd[0]
        rdm2s[0, i, :, :, i] -= rdm1s_2nd[0]
        rdm2s[0, :, i, i, :] -= rdm1s_2nd[0]
        rdm2s[0, :, :, i, i] += rdm1s_2nd[0]
        # mixed alpha-beta
        rdm2s[1, i, i, :, :] += rdm1s_2nd[1]
    for i in range(noccb):
        # mixed alpha-beta
        rdm2s[1, :, :, i, i] += rdm1s_2nd[0]
        # pure beta contributions
        rdm2s[2, i, i, :, :] += rdm1s_2nd[1]
        rdm2s[2, i, :, :, i] -= rdm1s_2nd[1]
        rdm2s[2, :, i, i, :] -= rdm1s_2nd[1]
        rdm2s[2, :, :, i, i] += rdm1s_2nd[1]

    return


def make_rdm2_tprod_ump2(
    nocca: int, noccb: int, t2aa: np.ndarray, t2ab: np.ndarray, t2bb: np.ndarray, rdm2s: np.ndarray
) -> None:
    """Second-order contributions to UHF-MP(2, 1) 2-RDM that are tensor products of MP2 amplitudes.

    Args:
        nocca:  Number of occupied alpha orbitals.
        noccb:  Number of occupied beta orbitals.
        t2aa:   MP2 amplitudes with alpha orbitals.
        t2ab:   MP2 amplitudes with mixed alpha and beta orbitals (abab).
        t2bb:   MP2 amplitudes with beta orbitals.
        rdm2s:  The 2-RDM contribution is added to this array.
    """
    # Contribution with four occupied indices rdm2[i, j, k, l] = <i+ k+ l j>.
    rdm2s[0, :nocca, :nocca, :nocca, :nocca] += 0.5 * np.einsum("ikab,jlab->ijkl", t2aa, t2aa)
    rdm2s[1, :nocca, :nocca, :noccb, :noccb] += np.einsum("ikab,jlab->ijkl", t2ab, t2ab)
    rdm2s[2, :noccb, :noccb, :noccb, :noccb] += 0.5 * np.einsum("ikab,jlab->ijkl", t2bb, t2bb)

    # Contribution with four virtual indices rdm2[a, b, c, d] = <a+ c+ d b>.
    rdm2s[0, nocca:, nocca:, nocca:, nocca:] += 0.5 * np.einsum("ijac,ijbd->abcd", t2aa, t2aa)
    rdm2s[1, nocca:, nocca:, noccb:, noccb:] += np.einsum("ijac,ijbd->abcd", t2ab, t2ab)
    rdm2s[2, noccb:, noccb:, noccb:, noccb:] += 0.5 * np.einsum("ijac,ijbd->abcd", t2bb, t2bb)

    # Mixed occupied-virtual contribution: pure alpha part.
    G2aijb = np.einsum("ikac,jkbc->aijb", t2aa, t2aa) + np.einsum("ikac,jkbc->aijb", t2ab, t2ab)
    rdm2s[0, nocca:, :nocca, :nocca, nocca:] += G2aijb
    rdm2s[0, :nocca, nocca:, nocca:, :nocca] += G2aijb.transpose((1, 0, 3, 2))
    rdm2s[0, nocca:, nocca:, :nocca, :nocca] -= G2aijb.transpose((0, 3, 2, 1))
    rdm2s[0, :nocca, :nocca, nocca:, nocca:] -= G2aijb.transpose((1, 2, 3, 0))

    # Mixed occupied-virtual contribution: mixed alpha-beta part.
    rdm2s[1, nocca:, nocca:, :noccb, :noccb] -= np.einsum("kiac,kjbc->abji", t2ab, t2ab)
    rdm2s[1, :nocca, :nocca, noccb:, noccb:] -= np.einsum("ikca,jkcb->ijba", t2ab, t2ab)
    rdm2s[1, nocca:, :nocca, :noccb, noccb:] += np.einsum(
        "ikac,kjcb->aijb", t2aa, t2ab
    ) + np.einsum("ikac,kjcb->aijb", t2ab, t2bb)
    rdm2s[1, :nocca, nocca:, noccb:, :noccb] += np.einsum(
        "ikac,kjcb->iabj", t2aa, t2ab
    ) + np.einsum("ikac,kjcb->iabj", t2ab, t2bb)

    # Mixed occupied-virtual contribution: pure beta part.
    G2aijb = np.einsum("ikac,jkbc->aijb", t2bb, t2bb) + np.einsum("kica,kjcb->aijb", t2ab, t2ab)
    rdm2s[2, noccb:, :noccb, :noccb, noccb:] += G2aijb
    rdm2s[2, :noccb, noccb:, noccb:, :noccb] += G2aijb.transpose((1, 0, 3, 2))
    rdm2s[2, noccb:, noccb:, :noccb, :noccb] -= G2aijb.transpose((0, 3, 2, 1))
    rdm2s[2, :noccb, :noccb, noccb:, noccb:] -= G2aijb.transpose((1, 2, 3, 0))


def make_rdm2_contractions_ump2(
    mol: Mole,
    nocca: int,
    noccb: int,
    mo_coeff: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    mo_energy: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    t2aa: np.ndarray,
    t2ab: np.ndarray,
    t2bb: np.ndarray,
    rdm2s: np.ndarray,
) -> None:
    """Second-order contributions to UHF-MP(2, 1) 2-RDM that are amplitude-integral contractions.

    Args:
        mol:        PySCF Mole object containing molecular data.
        nocca:      Number of occupied alpha orbitals.
        noccb:      Number of occupied beta orbitals.
        mo_coeff:   Molecular orbital coefficients.
        mo_energy:  Molecular orbital energies.
        t2aa:       Pure alpha MP2 amplitudes.
        t2ab:       Mixed alpha-beta MP2 amplitudes.
        t2bb:       Pure beta MP2 amplitudes.
        rdm2s:      The 2-RDM contribution is added to this array.

    Raises:
        ValueError: Invalid input.
    """
    Ca = mo_coeff[0]
    Cb = mo_coeff[1]
    nmo = Ca.shape[1]
    if Cb.shape[1] != nmo:
        raise ValueError("Inconsistent numbers of alpha and beta orbitals.")

    # Calculate ERIs.
    eri_aa = ao2mo.full(mol, Ca, aosym=1).reshape((nmo, nmo, nmo, nmo))
    eri_ab = ao2mo.general(mol, (Ca, Ca, Cb, Cb), aosym=1).reshape((nmo, nmo, nmo, nmo))
    eri_bb = ao2mo.full(mol, Cb, aosym=1).reshape((nmo, nmo, nmo, nmo))

    # Mixed occupied virtual part: not decomposable into a tensor product of amplitudes.
    # Pure alpha part.
    numerator = np.einsum("ajkc,ikbc->aibj", eri_aa[nocca:, :nocca, :nocca, nocca:], t2aa)
    numerator -= np.einsum("ackj,ikbc->aibj", eri_aa[nocca:, nocca:, :nocca, :nocca], t2aa)
    numerator += np.einsum("ajkc,ikbc->aibj", eri_ab[nocca:, :nocca, :noccb, noccb:], t2ab)
    # Antisymmetrizing terms from the preceding three lines.
    numerator = numerator - numerator.transpose((2, 1, 0, 3))
    numerator = numerator - numerator.transpose((0, 3, 2, 1))
    # Further terms in the numerator that do not need to be antisymmetrized.
    numerator -= np.einsum("acbd,ijcd->aibj", eri_aa[nocca:, nocca:, nocca:, nocca:], t2aa)
    numerator -= np.einsum("ikjl,klab->aibj", eri_aa[:nocca, :nocca, :nocca, :nocca], t2aa)
    # Denominator consists of MO energy differences (e_a + e_b - e_i - e_j).
    denominator = np.zeros((nmo - nocca, nocca, nmo - nocca, nocca))
    for a in range(nocca, nmo):
        denominator[a - nocca, :, :, :] += mo_energy[0][a]
        denominator[:, :, a - nocca, :] += mo_energy[0][a]
    for i in range(nocca):
        denominator[:, i, :, :] -= mo_energy[0][i]
        denominator[:, :, :, i] -= mo_energy[0][i]
    # Add contribution to the 2-RDM.
    G2aibj = numerator / denominator
    rdm2s[0, nocca:, :nocca, nocca:, :nocca] += G2aibj
    rdm2s[0, :nocca, nocca:, :nocca, nocca:] += G2aibj.transpose((1, 0, 3, 2))

    # Mixed occupied virtual part: not decomposable into a tensor product of amplitudes.
    # Mixed alpha-beta part.
    # Disentangling permutations and spin cases.
    numerator = np.einsum("ackj,ikcb->aibj", eri_ab[nocca:, nocca:, :noccb, :noccb], t2ab)
    numerator -= np.einsum("aikc,kjcb->aibj", eri_aa[nocca:, :nocca, :nocca, nocca:], t2ab)
    numerator += np.einsum("acki,kjcb->aibj", eri_aa[nocca:, nocca:, :nocca, :nocca], t2ab)
    numerator -= np.einsum("aikc,jkbc->aibj", eri_ab[nocca:, :nocca, :noccb, noccb:], t2bb)
    numerator -= np.einsum("kcbj,ikac->aibj", eri_ab[:nocca, nocca:, noccb:, :noccb], t2aa)
    numerator -= np.einsum("bjkc,ikac->aibj", eri_bb[noccb:, :noccb, :noccb, noccb:], t2ab)
    numerator += np.einsum("bckj,ikac->aibj", eri_bb[noccb:, noccb:, :noccb, :noccb], t2ab)
    numerator += np.einsum("kibc,kjac->aibj", eri_ab[:nocca, :nocca, noccb:, noccb:], t2ab)
    # Terms in the numerator without permutations.
    numerator -= np.einsum("acbd,ijcd->aibj", eri_ab[nocca:, nocca:, noccb:, noccb:], t2ab)
    numerator -= np.einsum("ikjl,klab->aibj", eri_ab[:nocca, :nocca, :noccb, :noccb], t2ab)
    # Denominator consists of MO energy differences (e_a + e_b - e_i - e_j).
    denominator = np.zeros((nmo - nocca, nocca, nmo - noccb, noccb))
    for a in range(nocca, nmo):
        denominator[a - nocca, :, :, :] += mo_energy[0][a]
    for b in range(noccb, nmo):
        denominator[:, :, b - noccb, :] += mo_energy[1][b]
    for i in range(nocca):
        denominator[:, i, :, :] -= mo_energy[0][i]
    for j in range(noccb):
        denominator[:, :, :, j] -= mo_energy[1][j]
    # Add contribution to the 2-RDM.
    G2aibj = numerator / denominator
    rdm2s[1, nocca:, :nocca, noccb:, :noccb] += G2aibj
    rdm2s[1, :nocca, nocca:, :noccb, noccb:] += G2aibj.transpose((1, 0, 3, 2))

    # Mixed occupied virtual part: not decomposable into a tensor product of amplitudes.
    # Pure beta part.
    numerator = np.einsum("ajkc,ikbc->aibj", eri_bb[noccb:, :noccb, :noccb, noccb:], t2bb)
    numerator -= np.einsum("ackj,ikbc->aibj", eri_bb[noccb:, noccb:, :noccb, :noccb], t2bb)
    numerator += np.einsum("kcaj,kicb->aibj", eri_ab[:nocca, nocca:, noccb:, :noccb], t2ab)
    # Antisymmetrizing terms from the preceding three lines.
    numerator = numerator - numerator.transpose((2, 1, 0, 3))
    numerator = numerator - numerator.transpose((0, 3, 2, 1))
    # Further terms in the numerator that do not need to be antisymmetrized.
    numerator -= np.einsum("acbd,ijcd->aibj", eri_bb[noccb:, noccb:, noccb:, noccb:], t2bb)
    numerator -= np.einsum("ikjl,klab->aibj", eri_bb[:noccb, :noccb, :noccb, :noccb], t2bb)
    # Denominator consists of MO energy differences (e_a + e_b - e_i - e_j).
    denominator = np.zeros((nmo - noccb, noccb, nmo - noccb, noccb))
    for a in range(noccb, nmo):
        denominator[a - noccb, :, :, :] += mo_energy[1][a]
        denominator[:, :, a - noccb, :] += mo_energy[1][a]
    for i in range(noccb):
        denominator[:, i, :, :] -= mo_energy[1][i]
        denominator[:, :, :, i] -= mo_energy[1][i]
    # Add contribution to the 2-RDM.
    G2aibj = numerator / denominator
    rdm2s[2, noccb:, :noccb, noccb:, :noccb] += G2aibj
    rdm2s[2, :noccb, noccb:, :noccb, noccb:] += G2aibj.transpose((1, 0, 3, 2))


def make_rdm2_rmp2(pt: RMP2class, mo_energy: np.ndarray) -> np.ndarray:
    """Calculate MP(2, 1) two-particle reduced density matrix (RHF reference).

    Args:
        pt:         Restricted MP2 object.
        mo_energy:  MO orbital energies as a vector.

    Returns:
        2-RDM in MO basis
    """
    # Molecular data.
    mol = pt.mol

    # Spin-free MP2 amplitudes: t^ij_ab -> t[i, j, a, b] with ij occupied, ab virtual
    t2 = pt.t2

    # Total number of MOs.
    nmo = pt.get_nmo()

    # Number of occupied MOs.
    nocc = pt.get_nocc()

    # 1-RDM in restricted MO basis.
    rdm1 = make_rdm1_rmp2(pt, mo_energy)

    # We will need the 2nd order components of the 1-RDM only. For this reason, the 0th order (HF)
    # component is subtracted, consisting of 2 * delta_ij for occupied orbitals.
    rdm1_2nd = rdm1.copy()
    for i in range(nocc):
        rdm1_2nd[i, i] -= 2.0

    # Calculate ERIs.
    eri = ao2mo.full(mol, pt.mo_coeff, aosym=1).reshape((nmo, nmo, nmo, nmo))

    # Initialize array for the 2-RDM
    rdm2 = np.zeros((nmo, nmo, nmo, nmo))

    # 0th order part of the 2-RDM: Hartree-Fock contribution.
    for i in range(nocc):
        for j in range(nocc):
            rdm2[i, i, j, j] += 4.0
            rdm2[i, j, j, i] -= 2.0

    # 1st-order part of the 2-RDM.
    t2c = 4.0 * t2 - 2.0 * t2.transpose((0, 1, 3, 2))
    rdm2[:nocc, nocc:, :nocc, nocc:] += t2c.transpose((0, 2, 1, 3))
    rdm2[nocc:, :nocc, nocc:, :nocc] += t2c.transpose((2, 0, 3, 1))

    #############################################################################
    # From this point until the end of the function: 2nd-order part of the 2-RDM.
    #############################################################################

    # Products of 2nd order 1-RDM with Kronecker deltas.
    for i in range(nocc):
        rdm2[i, i, :, :] += 2.0 * rdm1_2nd
        rdm2[i, :, :, i] -= rdm1_2nd
        rdm2[:, i, i, :] -= rdm1_2nd
        rdm2[:, :, i, i] += 2.0 * rdm1_2nd

    # Contribution with four occupied indices rdm2[i, j, k, l].
    rdm2[:nocc, :nocc, :nocc, :nocc] += np.einsum("ikab,jlab->ijkl", t2, t2c)

    # Contribution with four virtual indices rdm2[a, b, c, d].
    rdm2[nocc:, nocc:, nocc:, nocc:] += np.einsum("ijac,ijbd->abcd", t2, t2c)

    # Mixed occupied-virtual contribution: tensor product of amplitudes
    G2abji = -np.einsum("ikac,jkbc->abji", t2, t2c) - np.einsum("ikca,jkcb->abji", t2, t2c)
    rdm2[nocc:, nocc:, :nocc, :nocc] += G2abji
    rdm2[:nocc, :nocc, nocc:, nocc:] += G2abji.transpose((2, 3, 0, 1))
    del G2abji
    G2aijb = 0.5 * np.einsum("ikac,jkbc->aijb", t2c, t2c)
    rdm2[nocc:, :nocc, :nocc, nocc:] += G2aijb
    rdm2[:nocc, nocc:, nocc:, :nocc] += G2aijb.transpose((2, 3, 0, 1))
    del G2aijb

    # Mixed occupied virtual part: not decomposable into a tensor product of amplitudes.
    numerator = -2.0 * np.einsum("aikc,jkbc->aibj", eri[nocc:, :nocc, :nocc, nocc:], t2c)
    numerator += np.einsum("acki,jkbc->aibj", eri[nocc:, nocc:, :nocc, :nocc], t2c)
    numerator += np.einsum("ajkc,ikbc->aibj", eri[nocc:, :nocc, :nocc, nocc:], t2c)
    numerator += np.einsum("ackj,ikcb->aibj", eri[nocc:, nocc:, :nocc, :nocc], t2c)
    numerator = numerator + numerator.transpose((2, 3, 0, 1))
    # Terms with purely virtual or occupied ERIs.
    numerator -= np.einsum("acbd,ijcd->aibj", eri[nocc:, nocc:, nocc:, nocc:], t2c)
    numerator -= np.einsum("ikjl,klab->aibj", eri[:nocc, :nocc, :nocc, :nocc], t2c)
    # Denominator consists of MO energy differences (e_a + e_b - e_i - e_j).
    denominator = np.zeros((nmo - nocc, nocc, nmo - nocc, nocc))
    for a in range(nocc, nmo):
        denominator[a - nocc, :, :, :] += mo_energy[a]
        denominator[:, :, a - nocc, :] += mo_energy[a]
    for i in range(nocc):
        denominator[:, i, :, :] -= mo_energy[i]
        denominator[:, :, :, i] -= mo_energy[i]
    # Add contribution to the 2-RDM.
    G2aibj = numerator / denominator
    rdm2[nocc:, :nocc, nocc:, :nocc] += G2aibj
    rdm2[:nocc, nocc:, :nocc, nocc:] += G2aibj.transpose((1, 0, 3, 2))

    return rdm2
