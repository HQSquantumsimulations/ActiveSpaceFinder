# Copyright © 2022 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Calculating pair information using Moller-Plesset theory with the RI approximation."""

from math import isclose
from typing import Union

import numpy as np
from pyscf import lib
from pyscf.mp.dfmp2_native import orbgrad_from_Gamma
from pyscf.mp.dfump2_native import DFUMP2, ump2_densities_contribs
from pyscf.scf.uhf import UHF

from .molmath import corresponding_orbitals, overlap_square_roots
from .pairinfo import transform_unrestricted_rdm1s


def space_with_threshold(
    reference: Union[UHF, DFUMP2],
    pair_thresh: float = 2.0e-3,
    svd_thresh: float = 0.98,
    active_mp2no: bool = True,
    inactive_mp2no: bool = True,
) -> tuple[int, list[int], np.ndarray]:
    """Construct a guess active space using pair information from perturbation theory.

    Args:
        reference:      Either a UHF (a DFUMP2 object is constructed) or an existing DFUMP2 object.
        pair_thresh:    Cutoff to truncate pair information.
        svd_thresh:     Cutoff to construct a minimal active space from UHF corresponding orbitals.
        active_mp2no:   Orbitals in minimal active space are MP2 natural orbital-like.
        inactive_mp2no: All other orbitals are MP2 natural orbital-like.

    Returns:
        An initial active space of orbitals similar to MP2 natural orbitals. It is provided as a
        tuple containing:
            - The number of electrons in the initial active space.
            - The list of MO indices for the initial active space.
            - Restricted MO coefficients that the MO list refers to.
    """
    from .preselection import MP2PairinfoPreselection

    pre = MP2PairinfoPreselection(
        reference=reference,
        pair_thresh=pair_thresh,
        svd_thresh=svd_thresh,
        active_mp2no=active_mp2no,
        inactive_mp2no=inactive_mp2no,
    )
    space = pre.select()
    return space.nel, space.mo_list, space.mo_coeff


def UHF_corresponding_orbitals(
    mo_coeff: Union[tuple[np.ndarray, np.ndarray], np.ndarray],
    nocc: Union[np.ndarray, tuple[int, int]],
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate corresponding orbitals between spin-unrestricted MOs.

    Occupied corresponding orbitals are defined via the SVD of the rectangular matrix of overlaps
    between occupied alpha and beta orbitals, S[i, j] = <i_alpha | j_beta>.
    Likewise, virtual corresponding orbitals are obtained through the SVD of the overlap matrix
    between virtual alpha and beta orbitals, S[a, b] = <a_alpha | b_beta>.
    Corresponding occupied and virtual orbitals of the same spin are combined into one matrix. The
    ordering of the MO coefficient columns are such that:
        1) Occupied orbitals come first / left, with descending singular values.
        2) Singular values of zero, with associated orbitals, come in the center.
        3) Virtual orbitals come last / right, with ascending singular values.
    The are at least as many singular values of zero as there are unpaired electrons.

    Args:
        mo_coeff:   Unrestricted molecular orbital coefficients.
        nocc:       Number of occupied orbitals (alpha, beta).
        S:          Overlap matrix between atomic basis functions.

    Returns:
        Tuple containing (singular values, corresponding orbital coefficients)

    Raises:
        ValueError: Invalid input.
    """
    # Sanity check of MO coefficients.
    mo_coeff = np.array(mo_coeff)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise ValueError("MO coefficients of dimensions 2 x N(AO) x N(MO).")

    # Total number of MOs.
    nmo = mo_coeff.shape[2]

    # Crop occupied alpha and beta MOs.
    mos_occ_a = mo_coeff[0, :, : nocc[0]]
    mos_occ_b = mo_coeff[1, :, : nocc[1]]
    # Overlap matrix between spatial parts of occupied alpha and beta MOs.
    Socc_ab = mos_occ_a.T @ S @ mos_occ_b
    # Corresponding orbitals: alpha coefficients, beta coefficients, singular values.
    Coa, Cob, sigma_o = corresponding_orbitals(Socc_ab, mos_occ_a, mos_occ_b)

    # Crop virtual alpha and beta MOs.
    mos_virt_a = mo_coeff[0][:, nocc[0] :]
    mos_virt_b = mo_coeff[1][:, nocc[1] :]
    # Overlap matrix between spatial parts of virtual alpha and beta MOs.
    Svirt_ab = mos_virt_a.T @ S @ mos_virt_b
    # Corresponding orbitals: alpha coefficients, beta coefficients, singular values.
    Cva, Cvb, sigma_v = corresponding_orbitals(Svirt_ab, mos_virt_a, mos_virt_b)

    # Combine alpha and beta corresponding orbitals: occupied by descending singular values,
    # virtual by ascending singular values.
    Ca = np.hstack((Coa, np.fliplr(Cva)))
    Cb = np.hstack((Cob, np.fliplr(Cvb)))
    # Collect alpha and beta MO coefficients in one array.
    C = np.array((Ca, Cb))

    # Combine singular values into one vector:
    #   1) Singular values for occupied orbitals in descending order.
    #   2) Values of zero to fill up.
    #   3) Singular values for virtual orbitals in ascending order.
    n_zeros = nmo - len(sigma_o) - len(sigma_v)
    sigma = np.concatenate((sigma_o, np.zeros(n_zeros), np.flip(sigma_v)))

    # Return tuple: (singular values, corresponding orbital coefficients)
    return (sigma, C)


def corresponding_orbital_subspaces(
    pt: DFUMP2, threshold: float = 0.98, active_mp2no: bool = True, inactive_mp2no: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs a transformation and partitioning of subspaces for RI-MP(2, 1) pair information.

    The orbital transformation is performed in two steps:
    1) Unrestricted corresponding orbitals are calculated for the unrestricted occupied and virtual
       spin orbitals, respectively.
    2) Pairs of occupied corresponding spin orbitals and pairs of virtual corresponding spin
       orbitals are selected for an initial "active space" if the associate singular value is below
       the specified threshold. Therefore, the resulting spin orbitals can be divided into four
       subspaces: occupied "inactive" MOs, occupied "active" MOs, virtual "active" MOs and virtual
       "inactive" MOs.
    3) Optionally, the spin orbitals are transformed further (switched on by default):
        - "Active" spin orbitals diagonalize the unrelaxed MP2 density matrix in their subspace.
          Separation of occupied and virtual orbitals is preserved.
        - "Inactive" spin orbitals are determined as a counterpart of corresponding orbitals, but
          with the spin-free unrelaxed MP2 density matrix as the metric instead of the overlap.
          Thus, they are similar to MP2 natural orbitals, but come in pairs of spin orbitals.
    4) Construct a set of restricted orbitals that the unrestricted orbitals can be mapped onto:
        - For "inactive" orbitals, a single restricted orbital is determined to be similar to each
          pair of alpha and beta orbitals.
        - For "active" orbitals, a corresponding orbital transformation of the entire active space
          is performed with overlap or with the MP2 density metric (mixing occupied and virtual
          MOs, but not active with inactive). These are used to determine restricted orbitals.

    Args:
        pt:             DF-UMP2 instance to perform MP2 calculations.
        threshold:      Threshold for singular values to distinguish identify active spin orbitals.
        active_mp2no:   Flag to diagonalize the MP2 density matrix for active spin orbitals.
        inactive_mp2no: Flag to perform SVD of the MP2 density matrix for inactive spin orbitals.

    Returns:
        Tuple containing (active indices, spin orbitals, restricted orbitals)
        - Active indices are in a single list, which applies to both alpha and beta orbitals.
        - Spin orbital coefficients stored as 2 x N(AO) x N(MO) array. The columns are ordered from
          left to right: occupied inactive, occupied active, virtual active, virtual inactive.
        - Restricted orbital coefficients stored as N(AO) x N(MO) matrix. Columns ordered from left
          to right: occupied inactive, active, virtual inactive.
    """
    # UHF MO coefficients.
    mo_scf = pt.mo_coeff
    # Number of occupied alpha and beta orbitals.
    nocc = pt.nocc

    # Overlap matrix of atomic basis functions.
    S = pt.mol.intor_symmetric("int1e_ovlp")

    # Calculate unrestricted corresponding orbitals.
    # sigma: singular values
    # mo_coeff: MO coefficients of the corresponding orbitals (alpha and beta)
    sigma, mo_corr = UHF_corresponding_orbitals(mo_scf, nocc, S)

    # Identify an initial minimal "active" space with singular values below the threshold.
    active = np.argwhere(sigma < threshold).flatten()
    inactive = np.argwhere(sigma >= threshold).flatten()

    # UHF-MP2 unrelaxed density matrix in AO basis: sum of alpha and beta components
    rdm1s_ao = pt.make_rdm1_unrelaxed(ao_repr=True)

    # Initialize the array of transformed MO coefficients as a copy of the corresponding orbitals.
    mo_coeff = np.array(mo_corr)

    # Indices of inactive MOs that are occupied.
    inactive_occ = inactive[inactive < min(nocc)]
    # Indices of inactive MOs that are virtual.
    inactive_virt = inactive[inactive >= max(nocc)]

    # Determining inactive spin orbitals.
    if inactive_mp2no:
        # Transform the spin-free UHF-MP2 into a mixed basis: rows represent the spatial components
        # of alpha orbitals, columns the spatial components of beta orbitals.
        # The purpose of this matrix is to construct corresponding orbitals, but with the 1-RDM as
        # a metric instead of the overlap matrix. Is it not a "mixed" RDM.
        rdm1ab = np.linalg.multi_dot([mo_corr[0].T, S, rdm1s_ao[0] + rdm1s_ao[1], S, mo_corr[1]])

        # Iterate over the occupied and virtual inactive subspaces.
        for mo_list in (inactive_occ, inactive_virt):
            # Cut out projected spin-free MP2 1-RDM for the subspace.
            rdm1ab_sub = rdm1ab[mo_list, :][:, mo_list]
            # Cut out MO coefficients for the subspace.
            moa_sub = mo_corr[0][:, mo_list]
            mob_sub = mo_corr[1][:, mo_list]
            # Corresponding orbital transformation with the MP2 1-RDM in place of the overlap
            # matrix. The MP2 1-RDM has been truncated to the orbital subspace to avoid mixing.
            moa_tf, mob_tf, _ = corresponding_orbitals(rdm1ab_sub, moa_sub, mob_sub)
            # Set the "corresponding" MO coefficients to the appropriate sub-block.
            mo_coeff[0][:, mo_list] = moa_tf
            mo_coeff[1][:, mo_list] = mob_tf

    # Determining active spin orbitals.
    if active_mp2no:
        # Iterate over alpha and beta spins.
        for s in (0, 1):
            # Indices of active MOs that are occupied.
            active_occ = active[active < nocc[s]]
            # Indices of active MOs that are virtual.
            active_virt = active[active >= nocc[s]]

            # Iterate over the occupied and virtual active subspaces.
            for mo_list in (active_occ, active_virt):
                # Cut out MO coefficients for the subspace.
                mos_sub = mo_corr[s][:, mo_list]
                # Relevant spin component of the MP2 1-RDM in the active subspace.
                rdm1s_sub = mos_sub.T @ S @ rdm1s_ao[s] @ S @ mos_sub
                # Diagonalize the projected 1-RDM block to obtain pseudo-natural spin orbitals.
                _, U = np.linalg.eigh(rdm1s_sub)
                # Set pseudo-natural spin orbitals, ordered by descending eigenvalue.
                mo_coeff[s][:, mo_list] = mos_sub @ np.fliplr(U)

    # Construct corresponding orbitals spanning active space (occupied and virtual combined):
    mo_act = mo_coeff[:, :, active]
    # unrelaxed MP2 1-RDM as the metric
    if active_mp2no:
        rdm1ab_act = mo_act[0].T @ S @ (rdm1s_ao[0] + rdm1s_ao[1]) @ S @ mo_act[1]
        coa_act, cob_act, _ = corresponding_orbitals(rdm1ab_act, mo_act[0], mo_act[1])
    # overlap matrix as the metric
    else:
        Sab_act = mo_act[0].T @ S @ mo_act[1]
        coa_act, cob_act, _ = corresponding_orbitals(Sab_act, mo_act[0], mo_act[1])

    # Combine three sets of corresponding orbitals into an array of spin orbitals:
    # 1) inactive occupied, active, inactive virtual
    co_for_re = np.concatenate(
        (
            mo_coeff[:, :, inactive_occ],
            np.array([coa_act, cob_act]),
            mo_coeff[:, :, inactive_virt],
        ),
        axis=2,
    )
    # Determine restricted orbitals for these corresponding orbitals.
    mo_restricted = ur_to_re_pairwise(co_for_re, S)

    # Return list of "active" orbitals (minimal space), transformed spin orbitals, restricted MOs.
    return (active, mo_coeff, mo_restricted)


def ur_to_re_pairwise(mo_coeff: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Constructs restricted orbitals from pairwise combination of unrestricted orbitals.

    The input MO coefficients are unrestricted orbitals with a one-to-one correspondence of their
    spatial parts; for example, if the orbitals were constructed via a corresponding orbital
    transformation of their respective occupied and virtual parts. Thus, it is expected that
    mo_coeff[0, :, i] and mo_coeff[1, :, i] have similar spatial parts. In the returned matrix,
    mo_orth[:, i] will contain a spin-restricted approximation of the two spin orbitals.

    Args:
        mo_coeff:   MO coefficients of the unrestricted orbitals (alpha, beta).
        S:          Overlap matrix of atomic basis functions.

    Returns:
        Orthonormal MO coefficients of the restricted orbitals.

    Raises:
        ValueError: Bad input.
    """
    # Sanity check of the MO coefficients.
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise ValueError("mo_coeff must consist of two matrices (shape 2 x n(AO) x n(MO)).")

    # Number of molecular orbitals.
    nmo = mo_coeff.shape[2]

    # Construct a non-orthogonal matrix of spin-restricted coefficients by adding their MO
    # coefficients. If the orbitals have different phase factors (overlap below zero), subtract
    # their coefficients instead of adding them.
    mo_alpha, mo_beta = mo_coeff
    mo_restr = np.zeros_like(mo_alpha)
    for i in range(nmo):
        sign = 1.0 if mo_alpha[:, i].T @ S @ mo_beta[:, i] >= 0.0 else -1.0
        mo_restr[:, i] = 0.5 * (mo_alpha[:, i] + sign * mo_beta[:, i])

    # Perform symmetric orthogonalization or the restricted orbitals.
    SMO = mo_restr.T @ S @ mo_restr
    _, Sm12 = overlap_square_roots(SMO)
    mo_orth = mo_restr @ Sm12

    # Returns the orthonormal restricted orbitals.
    return mo_orth


def make_rdm1_ump2(pt: DFUMP2) -> np.ndarray:
    """Calculate modified RI-MP2 one-particle reduced density matrix (UHF reference).

    Args:
        pt:         Unrestricted MP2 object as a native DF-MP2 instance from PySCF.

    Returns:
        1-RDM in MO basis containing the alpha and beta components.

    Raises:
        ValueError: Invalid input.
    """
    # Make sure that RI integrals have been calculated
    if not pt.has_ints:
        pt.calculate_integrals_()
    intsfiles = pt._intsfile

    # Retrieve data from the MP2 object. No frozen core or spin-component scaling supported.
    mo_energy = pt.mo_energy
    frozen_mask = pt.frozen_mask
    if frozen_mask.any():
        raise ValueError("Frozen core orbitals not supported")
    nocc = pt.nocc
    nmo = pt.nmo
    mo_coeff = pt.mo_coeff
    max_memory = pt.max_memory
    mol = pt.mol
    auxmol = pt.auxmol
    if not isclose(pt.ps, 1.0) or not isclose(pt.pt, 1.0):
        raise ValueError("Spin-component scaling not supported")

    # The functions called require a PySCF logger object to be passed.
    logger = lib.logger.new_logger(pt)

    # Calculate:
    # (1) Unrelaxed MP2 difference 1-RDM (without SCF contribution), stored in rdm1.
    # (2) Two-electron-three-index RI density, stored in rdm2e3c (H5TmpFile object of PySCF).
    calcGamma = True
    rdm1, rdm2e3c = ump2_densities_contribs(
        intsfiles, mo_energy, frozen_mask, max_memory, logger, calcGamma, auxmol
    )

    # Iterate over spin-up and spin-down contributions.
    for s, Gset in [(0, rdm2e3c["Gamma_alpha"]), (1, rdm2e3c["Gamma_beta"])]:
        # Add SCF contributions to the 1-RDM.
        for i in range(nocc[s]):
            rdm1[s, i, i] += 1.0

        # The occ.-virt. block of the 1-RDM happens to equal the orbital gradient of the
        # two-electron term in the Hylleraas functional, divided by twice the difference of the
        # orbital energies.
        Lvo, _ = orbgrad_from_Gamma(
            mol, auxmol, Gset, mo_coeff[s], frozen_mask[s], max_memory, logger
        )
        for i in range(nocc[s]):
            for a in range(nocc[s], nmo):
                Dai = 0.5 * Lvo[a - nocc[s], i] / (mo_energy[s, a] - mo_energy[s, i])
                rdm1[s, a, i] += Dai
                rdm1[s, i, a] += Dai

    return rdm1


# Overlap threshold to detect mixing between orbital blocks.
default_overlap_thresh = 1e-10


def overlap_blocks(
    pt: DFUMP2, mo_coeff: np.ndarray, overlap_thresh: float = default_overlap_thresh
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Obtain transformation matrices of occupied and virtual spin orbital blocks separately.

    The original MOs in pt are represented in the rows of the transformation matrices, the
    transformed basis mo_coeff is represented in the respective columns.

    Args:
        pt:             DFUMP2 object containing the original MO coefficients.
        mo_coeff:       The transformed MO coefficients (must not mix occupied and virtual MOs).
        overlap_thresh: Threshold to determine that MO blocks do not get mixed.

    Returns:
        Transformation matrices U between the old and new MO coefficients as nested tuples.
        ((U for occupied alpha, U for occupied beta), (U for virtual alpha, U for virtual beta))

    Raises:
        ValueError:     Mixing of occupied and virtual MOs.
    """
    # Number of MOs (total, occupied and virtual).
    nocc = pt.nocc

    # Overlap matrix of atomic basis functions.
    S = pt.mol.intor_symmetric("int1e_ovlp")

    # Separate transformation matrices for occupied and virtual spin orbitals.
    # Rows of each matrix: original MOs in pt.mo_coeff. Columns: rotated MOs in mo_coeff.
    Uoo = []
    Uvv = []
    for s in (0, 1):
        U = lib.dot(lib.dot(pt.mo_coeff[s].T, S), mo_coeff[s])
        Uov = U[: nocc[s], nocc[s] :]
        Uvo = U[nocc[s] :, : nocc[s]]

        # Check that the MO blocks do not get mixed.
        if np.any(np.linalg.norm(Uov, axis=0) > overlap_thresh):
            mixing = True
        elif np.any(np.linalg.norm(Uov, axis=1) > overlap_thresh):
            mixing = True
        elif np.any(np.linalg.norm(Uvo, axis=0) > overlap_thresh):
            mixing = True
        elif np.any(np.linalg.norm(Uvo, axis=1) > overlap_thresh):
            mixing = True
        else:
            mixing = False
        if mixing:
            raise ValueError(f"mo_coeff mixing occupied and virtual spin orbitals for spin {s}.")

        Uoo.append(U[: nocc[s], : nocc[s]])
        Uvv.append(U[nocc[s] :, nocc[s] :])

    return ((Uoo[0], Uoo[1]), (Uvv[0], Uvv[1]))


def make_rdm2diag_0th_ump2(pt: DFUMP2, mo_coeff: np.ndarray) -> np.ndarray:
    """SCF (0th order) contribution to diagonal 2-RDM elements from UHF in a rotated basis.

    Args:
        pt:         DFUMP2 object
        mo_coeff:   MO coefficients of the target spin orbitals (not the SCF orbitals).

    Returns:
        Contributions to the "diagonal" elements [p, p, q, q] of the 2-RDM in a target orbital
        basis defined by the MO coefficients in mo_coeff. The result is an array of dimensions
        3 x N(MO) x N(MO), containing alpha-alpha, alpha-beta and beta-beta contributions.
    """
    nmo = pt.nmo
    nocca, noccb = pt.nocc
    rdm1a_hf = np.zeros((nmo, nmo))
    rdm1b_hf = np.zeros((nmo, nmo))
    rdm1a_hf[np.arange(nocca), np.arange(nocca)] = 1.0
    rdm1b_hf[np.arange(noccb), np.arange(noccb)] = 1.0
    rdm1s_tf = transform_unrestricted_rdm1s(pt.mo_coeff, mo_coeff, pt.mol, (rdm1a_hf, rdm1b_hf))
    rdm1a_tf, rdm1b_tf = rdm1s_tf
    rdm2_diag = np.zeros((3, nmo, nmo))
    rdm2_diag[0] = np.outer(np.diag(rdm1a_tf), np.diag(rdm1a_tf)) - rdm1a_tf**2
    rdm2_diag[1] = np.outer(np.diag(rdm1a_tf), np.diag(rdm1b_tf))
    rdm2_diag[2] = np.outer(np.diag(rdm1b_tf), np.diag(rdm1b_tf)) - rdm1b_tf**2
    return rdm2_diag


def make_rdm2diag_separable_ump2(
    pt: DFUMP2, rdm1s_mp2: np.ndarray, mo_coeff: np.ndarray
) -> np.ndarray:
    """Separable contributions to diagonal elements of the UHF-MP2(2, 1) 2-RDM.

    This function calculates contributions that can be separated into products of 1-RDMs.

    Args:
        pt:         DFUMP2 object containing data to perform RI-MP2 calculations.
        rdm1s_mp2:  The complete MP2 1-RDM in UHF orbital basis: can be the unrelaxed 1-RDM if
                    there is no mixing of occupied and virtual spin orbitals. With mixing, it must
                    be the MP(2, 1) 1-RDM, as calculated with make_rdm1_ump2(...).
        mo_coeff:   MO coefficients of the target spin orbitals (not the SCF orbitals).

    Returns:
        Contributions to the "diagonal" elements [p, p, q, q] of the 2-RDM in a target orbital
        basis defined by the MO coefficients in mo_coeff. The result is an array of dimensions
        3 x N(MO) x N(MO), containing alpha-alpha, alpha-beta and beta-beta contributions.

    Raises:
        ValueError: Invalid input.
    """
    # Number of all MOs.
    nmo = pt.nmo
    # Sanity check on the input arrays.
    if rdm1s_mp2.shape != (2, nmo, nmo):
        raise ValueError(f"1-RDM (rdm1s) with bad shape: {rdm1s_mp2.shape}")
    if mo_coeff.shape != (2, pt.mol.nao, nmo):
        raise ValueError(f"Target MOs with bad shape: {mo_coeff.shape}")

    # The SCF spin density matrix in MO basis.
    rdm1s_scf = np.zeros((2, nmo, nmo))
    rdm1s_scf[0, np.arange(pt.nocc[0]), np.arange(pt.nocc[0])] = 1
    rdm1s_scf[1, np.arange(pt.nocc[1]), np.arange(pt.nocc[1])] = 1

    # U is the transformation matrix from the original SCF orbitals to the target orbitals.
    S = pt.mol.intor_symmetric("int1e_ovlp")
    U = np.array([pt.mo_coeff[s].T @ S @ mo_coeff[s] for s in (0, 1)])

    # We will only need the second-order part of the MP density here.
    rdm1s_2nd = rdm1s_mp2 - rdm1s_scf

    # Transform the MP and SCF 1-RDMs to the target spin orbitals.
    rdm1s_2nd_transformed = np.array([U[s].T @ rdm1s_2nd[s] @ U[s] for s in (0, 1)])
    rdm1s_scf_transformed = np.array([U[s].T @ rdm1s_scf[s] @ U[s] for s in (0, 1)])

    # Array to store the result: alpha-alpha, alpha-beta and beta-beta (in that order).
    rdm2_diag = np.zeros((3, nmo, nmo))

    # alpha-alpha part
    rdm2_diag[0] = np.outer(np.diag(rdm1s_2nd_transformed[0]), np.diag(rdm1s_scf_transformed[0]))
    rdm2_diag[0] = rdm2_diag[0] + rdm2_diag[0].T
    rdm2_diag[0] -= 2.0 * rdm1s_2nd_transformed[0] * rdm1s_scf_transformed[0]

    # alpha-beta part
    rdm2_diag[1] += np.outer(np.diag(rdm1s_2nd_transformed[0]), np.diag(rdm1s_scf_transformed[1]))
    rdm2_diag[1] += np.outer(np.diag(rdm1s_scf_transformed[0]), np.diag(rdm1s_2nd_transformed[1]))

    # beta-beta part
    rdm2_diag[2] = np.outer(np.diag(rdm1s_2nd_transformed[1]), np.diag(rdm1s_scf_transformed[1]))
    rdm2_diag[2] = rdm2_diag[2] + rdm2_diag[2].T
    rdm2_diag[2] -= 2.0 * rdm1s_2nd_transformed[1] * rdm1s_scf_transformed[1]

    return rdm2_diag


def make_rdm2diag_vvvv_ump2(
    pt: DFUMP2, mo_coeff: np.ndarray, overlap_thresh: float = default_overlap_thresh
) -> np.ndarray:
    """All-virtual contributions to diagonal elements of the UHF-MP(2, 1) 2-RDM.

    The function requires that the occupied and virtual spin orbitals do not mix.

    Args:
        pt:             DFUMP2 object containing data to perform RI-MP2 calculations.
        mo_coeff:       MO coefficients of the target spin orbitals (preserving occupied-virtual
                        separation).
        overlap_thresh: Threshold to determine occupied-virtual MO mixing.

    Returns:
        Contributions to the "diagonal" elements [p, p, q, q] of the 2-RDM in a target orbital
        basis defined by the MO coefficients in mo_coeff. The result is an array of dimensions
        3 x N(MO) x N(MO), containing alpha-alpha, alpha-beta and beta-beta contributions.
    """
    # Number of all MOs.
    nmo = pt.nmo
    # Numbers of occupied alpha MOs and beta MOs.
    nocc = pt.nocc
    # Numbers of virtual alpha MOs and beta MOs.
    nvirt = nmo - nocc
    # Vectors of alpha and beta MO energies.
    mo_energy = pt.mo_energy

    # HDF5 data sets with the density-fitted three-center integrals.
    ints3c = tuple(pt._intsfile[s]["ints_cholesky"] for s in (0, 1))

    # Initializing the tensor to store diagonal 2-RDM elements.
    rdm2_diag = np.zeros((3, nmo, nmo))

    # Calculate the transformation matrix from canonical to target virtual spin orbitals.
    _, Uvirt = overlap_blocks(pt, mo_coeff, overlap_thresh)

    # Pure-spin contributions (alpha, beta).
    for s in (0, 1):
        # Sums of virtual MO energies to perform division through matrix operations.
        Eab = np.tile(mo_energy[s, nocc[s] :, np.newaxis], reps=(1, nvirt[s]))
        Eab += np.tile(mo_energy[s, np.newaxis, nocc[s] :], reps=(nvirt[s], 1))

        # Iterate over pairs of occupied spin orbitals of the same spin.
        for i in range(nocc[s]):
            ints3c_i = ints3c[s][i, :, :]
            for j in range(i):
                ints3c_j = ints3c[s][j, :, :]
                # Two-electron integrals in MO basis.
                Kab = lib.dot(ints3c_i.T, ints3c_j)
                # Energy demoninator (occupied minus virtual).
                DE = (mo_energy[s, i] + mo_energy[s, j]) - Eab
                # The amplitudes
                Tab = (Kab - Kab.T) / DE
                # Transform the amplitudes to the virtual target basis.
                T_transformed = lib.dot(lib.dot(Uvirt[s].T, Tab), Uvirt[s])
                # Adding diagonal 2-RDM contribution.
                rdm2_diag[2 * s, nocc[s] :, nocc[s] :] += T_transformed**2

    # Sums of virtual MO energies to perform division through matrix operations.
    Eab = np.tile(mo_energy[0, nocc[0] :, np.newaxis], reps=(1, nvirt[1]))
    Eab += np.tile(mo_energy[1, np.newaxis, nocc[1] :], reps=(nvirt[0], 1))

    # Iterations over opposite-spin pairs (i: alpha, j: beta)
    for i in range(nocc[0]):
        ints3c_i = ints3c[0][i, :, :]
        for j in range(nocc[1]):
            ints3c_j = ints3c[1][j, :, :]
            # Two-electron integrals in MO basis.
            Kab = lib.dot(ints3c_i.T, ints3c_j)
            # Energy demoninator (occupied minus virtual).
            DE = (mo_energy[0, i] + mo_energy[1, j]) - Eab
            # The amplitudes
            Tab = Kab / DE
            # Transform the amplitudes to the virtual target basis.
            T_transformed = lib.dot(lib.dot(Uvirt[0].T, Tab), Uvirt[1])
            # Adding diagonal 2-RDM contribution.
            rdm2_diag[1, nocc[0] :, nocc[1] :] += T_transformed**2

    return rdm2_diag


def make_rdm2diag_oooo_ump2(
    pt: DFUMP2, mo_coeff: np.ndarray, overlap_thresh: float = default_overlap_thresh
) -> np.ndarray:
    """All-occupied contributions to diagonal elements of the UHF-MP(2, 1) 2-RDM.

    This function calculates the contributions originating from products of amplitudes. It requires
    that the occupied and virtual spin orbitals do not mix.

    Args:
        pt:             DFUMP2 object containing data to perform RI-MP2 calculations.
        mo_coeff:       MO coefficients of the target spin orbitals (preserving occupied-virtual
                        separation).
        overlap_thresh: Threshold to determine occupied-virtual MO mixing.

    Returns:
        Contributions to the "diagonal" elements [p, p, q, q] of the 2-RDM in a target orbital
        basis defined by the MO coefficients in mo_coeff. The result is an array of dimensions
        3 x N(MO) x N(MO), containing alpha-alpha, alpha-beta and beta-beta contributions.

    Raises:
        MemoryError:    Insufficient memory to buffer RI integrals in memory.
    """
    # Number of all MOs.
    nmo = pt.nmo
    # Numbers of occupied alpha MOs and beta MOs.
    nocc = pt.nocc
    # Numbers of virtual alpha MOs and beta MOs.
    nvirt = nmo - nocc
    # Number of auxiliary functions.
    naux = pt.auxmol.nao
    # Vectors of alpha and beta MO energies.
    mo_energy = pt.mo_energy
    # Upper limit on the memory in MB.
    max_memory = pt.max_memory

    # HDF5 data sets with the density-fitted three-center integrals.
    ints3c = tuple(pt._intsfile[s]["ints_cholesky"] for s in (0, 1))

    # Initializing the tensor to store diagonal 2-RDM elements.
    rdm2_diag = np.zeros((3, nmo, nmo))

    # Calculate the transformation matrix from canonical to target occupied spin orbitals.
    Uocc, _ = overlap_blocks(pt, mo_coeff, overlap_thresh)

    # Pure-spin contributions (alpha, beta).
    for s in 0, 1:
        # Sums of occupied MO energies to perform division through matrix operations.
        Eij = np.tile(mo_energy[s, : nocc[s], np.newaxis], reps=(1, nocc[s]))
        Eij += np.tile(mo_energy[s, np.newaxis, : nocc[s]], reps=(nocc[s], 1))

        # This function requires sets of RI integrals with all occupied and auxiliary indices, but
        # only a single virtual index. Reading these integrals from disk results in fragmented,
        # slow operations. Here, we try to read integrals for as many virtual orbitals as possible
        # at the same time.
        # The buffer size is determined by the size of a matrix of RI integrals for a single
        # virtual index, 8 * nocc[s] * naux. We need to buffers for two virtual indices (8 -> 16).
        bufsize = int(1e6 * (max_memory - lib.current_memory()[0]) / (16 * nocc[s] * naux))
        if bufsize < 1:
            raise MemoryError("Insufficient memory (PYSCF_MAX_MEMORY).")

        # Iterate over batches of virtual orbitals for index a.
        for astart in range(0, nvirt[s], bufsize):
            astop = min(astart + bufsize, nvirt[s])

            # Read batch of integrals for selected virtual MOs from disk into memory.
            ints3c_abuf = ints3c[s][:, :, astart:astop]

            # Iterate over batches of virtual orbitals for index b.
            for bstart in range(0, astop, bufsize):
                bstop = min(bstart + bufsize, astop)

                # Read batch of integrals for selected virtual MOs from disk into memory.
                ints3c_bbuf = ints3c[s][:, :, bstart:bstop]

                # Iterate over pairs of virtual orbitals in the two batches.
                for a in range(astart, astop):
                    ints3c_a = ints3c_abuf[:, :, a - astart]
                    for b in range(bstart, min(bstop, a)):
                        ints3c_b = ints3c_bbuf[:, :, b - bstart]
                        # Two-electron integrals in MO basis.
                        Kij = lib.dot(ints3c_a, ints3c_b.T)
                        # Energy demoninator (occupied minus virtual).
                        DE = Eij - (mo_energy[s, nocc[s] + a] + mo_energy[s, nocc[s] + b])
                        # The amplitudes
                        Tij = (Kij - Kij.T) / DE
                        # Transform the amplitudes to the occupied target basis.
                        T_transformed = lib.dot(lib.dot(Uocc[s].T, Tij), Uocc[s])
                        # Adding diagonal 2-RDM contribution.
                        rdm2_diag[2 * s, : nocc[s], : nocc[s]] += T_transformed**2

        # Releasing memory for the buffering algorithm.
        del ints3c_abuf, ints3c_bbuf, ints3c_a, ints3c_b, Kij, DE, Tij, T_transformed

    # Sums of occupied MO energies to perform division through matrix operations.
    Eij = np.tile(mo_energy[0, : nocc[0], np.newaxis], reps=(1, nocc[1]))
    Eij += np.tile(mo_energy[1, np.newaxis, : nocc[1]], reps=(nocc[0], 1))

    # Sizes of two buffers to streamline the reading of integrals from disk, as above.
    bufsize1 = int(1e6 * (max_memory - lib.current_memory()[0]) / (16 * nocc[0] * naux))
    bufsize2 = int(1e6 * (max_memory - lib.current_memory()[0]) / (16 * nocc[1] * naux))
    if bufsize < 1 or bufsize2 < 1:
        raise MemoryError("Insufficient memory (PYSCF_MAX_MEMORY).")

    # Iterate over batches of virtual orbitals for index a.
    for astart in range(0, nvirt[0], bufsize1):
        astop = min(astart + bufsize1, nvirt[0])

        # Read batch of integrals for selected virtual MOs from disk into memory.
        ints3c_abuf = ints3c[0][:, :, astart:astop]

        # Iterate over batches of virtual orbitals for index b.
        for bstart in range(0, nvirt[1], bufsize2):
            bstop = min(bstart + bufsize2, nvirt[1])

            # Read batch of integrals for selected virtual MOs from disk into memory.
            ints3c_bbuf = ints3c[1][:, :, bstart:bstop]

            # Iterate over pairs of virtual orbitals in the two batches.
            for a in range(astart, astop):
                ints3c_a = ints3c_abuf[:, :, a - astart]
                for b in range(bstart, bstop):
                    ints3c_b = ints3c_bbuf[:, :, b - bstop]
                    # Two-electron integrals in MO basis.
                    Kij = lib.dot(ints3c_a, ints3c_b.T)
                    # Energy demoninator (occupied minus virtual).
                    DE = Eij - (mo_energy[0, nocc[0] + a] + mo_energy[1, nocc[1] + b])
                    # The amplitudes
                    Tij = Kij / DE
                    # Transform the amplitudes to the occupied target basis.
                    T_transformed = lib.dot(lib.dot(Uocc[0].T, Tij), Uocc[1])
                    # Adding diagonal 2-RDM contribution.
                    rdm2_diag[1, : nocc[0], : nocc[1]] += T_transformed**2

    # Releasing memory for the buffering algorithm (not needed here, but if someone moves code...)
    del ints3c_abuf, ints3c_bbuf, ints3c_a, ints3c_b, Kij, DE, Tij, T_transformed

    return rdm2_diag


def make_rdm2diag_oovv_ump2(
    pt: DFUMP2, mo_coeff: np.ndarray, overlap_thresh: float = default_overlap_thresh
) -> np.ndarray:
    """Mixed occupied-virtual contributions to diagonal elements of the UHF-MP(2, 1) 2-RDM.

    This function calculates the contributions originating from products of amplitudes. It requires
    that the occupied and virtual spin orbitals do not mix.

    Args:
        pt:             DFUMP2 object containing data to perform RI-MP2 calculations.
        mo_coeff:       MO coefficients of the target spin orbitals (preserving occupied-virtual
                        separation).
        overlap_thresh: Threshold to determine occupied-virtual MO mixing.

    Returns:
        Contributions to the "diagonal" elements [p, p, q, q] of the 2-RDM in a target orbital
        basis defined by the MO coefficients in mo_coeff. The result is an array of dimensions
        3 x N(MO) x N(MO), containing alpha-alpha, alpha-beta and beta-beta contributions.

    Raises:
        MemoryError:    Insufficient memory to buffer RI integrals in memory.
    """
    # Number of all MOs.
    nmo = pt.nmo
    # Numbers of occupied alpha MOs and beta MOs.
    nocc = pt.nocc
    # Numbers of virtual alpha MOs and beta MOs.
    nvirt = nmo - nocc
    # Vectors of alpha and beta MO energies.
    mo_energy = pt.mo_energy
    # Upper limit on the memory in MB.
    max_memory = pt.max_memory

    # HDF5 data sets with the density-fitted three-center integrals.
    ints3c = tuple(pt._intsfile[s]["ints_cholesky"] for s in (0, 1))

    # Separate transformation matrices for occupied and virtual spin orbitals.
    # Rows of each matrix: original MOs in pt.mo_coeff. Columns: rotated MOs in mo_coeff.
    Uo, Uv = overlap_blocks(pt, mo_coeff, overlap_thresh)

    # Array to store the result: alpha-alpha, alpha-beta and beta-beta (in that order).
    rdm2_diag = np.zeros((3, nmo, nmo))

    # Iteration over spin combinations.
    for s1, s2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # For each occupied spin orbital i, all amplitudes are calculated once and stored on disk
        # in the temporary file tfile. A batched algorithm is used to read the amplitudes back into
        # memory for processing. More memory -> more efficient I/O. The amplitudes are deleted
        # before proceeding to the next i. Thus, the storage scaling is O(N³).
        with lib.H5TmpFile(libver="latest") as tfile:
            # For fixed occupied MO i with spin s1, amplitudes are calculated with indices
            # j (occ., s2), a (virt., s1) and b (virt., s2) and stored in this data set.
            tiset = tfile.create_dataset("tjab", (nocc[s2], nvirt[s1], nvirt[s2]), dtype="f8")

            # Precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for division with numpy.
            Eab = np.tile(mo_energy[s1, nocc[s1] :, np.newaxis], reps=(1, nvirt[s2]))
            Eab += np.tile(mo_energy[s2, np.newaxis, nocc[s2] :], reps=(nvirt[s1], 1))

            # Iterate over fixed indices i.
            for i in range(nocc[s1]):
                ints3c_ia = ints3c[s1][i, :, :]

                # Calculate amplitudes for all indices j.
                for j in range(nocc[s2]):
                    ints3c_jb = ints3c[s2][j, :, :]
                    Kab = lib.dot(ints3c_ia.T, ints3c_jb)
                    DE = (mo_energy[s1, i] + mo_energy[s2, j]) - Eab
                    if s1 == s2:
                        Tab = (Kab - Kab.T) / DE
                    else:
                        Tab = Kab / DE
                    # Store in the disk buffer.
                    tiset[j, :, :] = Tab

                # Releasing memory for the buffering algorithm.
                del ints3c_ia, ints3c_jb, Kab, DE, Tab

                # batchsize is the maximal number of matrices that can be read into memory.
                avail_mb = max_memory - lib.current_memory()[0]
                batchsize = int(1e6 * avail_mb / (8 * nocc[s2] * nvirt[s2]))
                if batchsize < 1:
                    raise MemoryError("Insufficient memory (PYSCF_MAX_MEMORY).")

                for astart in range(0, nvirt[s1], batchsize):
                    astop = min(astart + batchsize, nvirt[s1])

                    # Read batch of amplitudes. More memory -> larger chunks astart:astop.
                    tbatch = tiset[:, astart:astop, :]
                    for a in range(astop - astart):
                        # Amplitudes with fixed i(s1), a(s1). Running indices j(s2), b(s2).
                        Tjb = tbatch[:, a, :]
                        # Transform the amplitudes to the target spin orbital basis.
                        Tjb_transformed = lib.dot(lib.dot(Uo[s2].T, Tjb), Uv[s2])
                        # Add contributions to diagonal elements of 2-RDM.
                        # Only occupied-virtual here. Virtual-occupied replicated at the bottom.
                        rdm2_diag[2 * s2, : nocc[s2], nocc[s2] :] -= Tjb_transformed**2

                # Memory management for the buffering algorithm.
                del tbatch, Tjb, Tjb_transformed

                # Extra contributions for amplitudes with mixed spin indices.
                if s1 != s2:
                    # Being pedantic about batch size.
                    avail_mb = max_memory - lib.current_memory()[0]
                    batchsize = int(1e6 * avail_mb / (8 * nocc[s2] * nvirt[s1]))
                    if batchsize < 1:
                        raise MemoryError("Insufficient memory (PYSCF_MAX_MEMORY).")

                    # Iterate over batches of virtual orbitals b with spin s2.
                    for bstart in range(0, nvirt[s2], batchsize):
                        bstop = min(bstart + batchsize, nvirt[s2])

                        # Read batch of amplitudes. More memory -> larger chunks bstart:bstop.
                        tbatch = tiset[:, :, bstart:bstop]
                        for b in range(bstop - bstart):
                            # Amplitudes with fixed i(s1), b(s2). Running indices j(s2), a(s1).
                            Tja = tbatch[:, :, b]
                            # Transform the amplitudes to the target spin orbital basis.
                            Tja_transformed = lib.dot(lib.dot(Uo[s2].T, Tja), Uv[s1])
                            # Add contributions to diagonal elements of 2-RDM.
                            if (s1, s2) == (0, 1):
                                rdm2_diag[1, nocc[s1] :, : nocc[s2]] -= Tja_transformed.T**2
                            else:  # (s1, s2) == (1, 0)
                                rdm2_diag[1, : nocc[s2], nocc[s1] :] -= Tja_transformed**2

                    # Memory management for the buffering algorithm.
                    del tbatch, Tja, Tja_transformed

    # Replicate occupied-virtual to virtual-occupied contributions.
    rdm2_diag[0, nocc[0] :, : nocc[0]] = rdm2_diag[0, : nocc[0], nocc[0] :].T
    rdm2_diag[2, nocc[1] :, : nocc[1]] = rdm2_diag[2, : nocc[1], nocc[1] :].T

    return rdm2_diag


def diag_cumulant_ump2(
    pt: DFUMP2, mo_coeff: np.ndarray, overlap_thresh: float = default_overlap_thresh
) -> np.ndarray:
    """Calculate diagonal elements of the two-particle-cumulant with UHF-RI-MP(2, 1).

    Diagonal cumulant elements are calculated in a rotated target MO basis, not necessarily in the
    UHF MO basis. The only requirement is that occupied and virtual spin orbitals are not mixed
    (unrelaxed natural spin orbitals are fine, spin-free natural orbitals in general are not).

    Args:
        pt:               DFUMP2 object containing data to perform RI-MP2 calculations.
        mo_coeff:         MO coefficients of the target spin orbitals (preserving occupied-virtual
                          separation).
        overlap_thresh:   Threshold to determine occupied-virtual MO mixing.

    Returns:
        Contributions to the "diagonal" elements [p, p, q, q] of the two-body cumulant in a target
        orbital basis defined by the MO coefficients in mo_coeff. The result is an array of
        dimensions 3 x N(MO) x N(MO), containing alpha-alpha, alpha-beta and beta-beta
        contributions.

    Raises:
        ValueError: Invalid input.
    """
    # Sanity checking.
    nmo = pt.nmo
    if mo_coeff.shape != (2, pt.mol.nao, nmo):
        raise ValueError(f"Target MOs with bad shape: {mo_coeff.shape}")

    # Calculate the MP(2, 1) 1-RDM.
    rdm1s_mp2 = make_rdm1_ump2(pt)

    rdm1a_tf, rdm1b_tf = transform_unrestricted_rdm1s(pt.mo_coeff, mo_coeff, pt.mol, rdm1s_mp2)

    # Zeroth order contribution from SCF 2-RDM.
    rdm2_diag = make_rdm2diag_0th_ump2(pt, mo_coeff)

    # Separable contribution to the 2-RDM. Since we require no occ.-virt. mixing, we can use either
    # this 1-RDM or the unrelaxed density matrix.
    rdm2_diag += make_rdm2diag_separable_ump2(pt, rdm1s_mp2, mo_coeff)

    # Contributions to the 2-RDM from products of amplitudes. Here, it is really important that
    # occupied and virtual orbitals do not get mixed.
    rdm2_diag += make_rdm2diag_vvvv_ump2(pt, mo_coeff, overlap_thresh)
    rdm2_diag += make_rdm2diag_oooo_ump2(pt, mo_coeff, overlap_thresh)
    rdm2_diag += make_rdm2diag_oovv_ump2(pt, mo_coeff, overlap_thresh)

    # Calculate the diagonal cumulant elements.
    cumulant = rdm2_diag
    cumulant[0] -= np.outer(np.diag(rdm1a_tf), np.diag(rdm1a_tf)) - rdm1a_tf**2
    cumulant[1] -= np.outer(np.diag(rdm1a_tf), np.diag(rdm1b_tf))
    cumulant[2] -= np.outer(np.diag(rdm1b_tf), np.diag(rdm1b_tf)) - rdm1b_tf**2
    return cumulant
