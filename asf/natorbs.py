# Copyright © 2020-2022 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Functions to calculate different types of natural orbitals."""

from typing import Optional, Sequence, Union

import numpy as np
from numpy.linalg import multi_dot
from pyscf import symm
from pyscf.cc.ccsd import CCSD as RCCSDClass
from pyscf.cc.uccsd import UCCSD as UCCSDClass
from pyscf.gto import Mole
from pyscf.mp.mp2 import RMP2 as RMP2Class
from pyscf.mp.ump2 import UMP2 as UMP2Class
from pyscf.scf.uhf import UHF as UHFClass
from scipy.linalg import eigh

from .molmath import overlap_square_roots
from .utility import iround

# Default eigenvalue cutoff to remove linear dependencies.
DEFAULT_STHRESH = 1.0e-8


def restricted_natural_orbitals(
    rdm1mo: np.ndarray, mo_coeff: np.ndarray, mol: Optional[Mole] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates natural orbitals from a density matrix in a basis of spin-restricted MOs.

    Args:
        rdm1mo:     one-particle reduced density matrix in MO basis
        mo_coeff:   molecular orbital coefficients
        mol:        Instance of pyscf.gto.Mole. Note that if mol.symmetry is enabled, the natural
                    orbitals will be symmetry adapted.

    Returns:
        natural occupation numbers, natural orbitals
    """
    if mol is not None and mol.symmetry:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)
        eigval, eigvec = symm.eigh(rdm1mo, orbsym)
    else:
        eigval, eigvec = eigh(rdm1mo)

    # The diagonalization routine orders the eigenvalue from lowest to highest.
    # We want to have the occupied orbitals first and the unoccupied ones last.
    natocc = np.flip(eigval)
    natorb = np.dot(mo_coeff, np.fliplr(eigvec))

    return natocc, natorb


def unrestricted_natural_orbitals(
    rdm1mo: tuple[np.ndarray, np.ndarray],
    mo_coeff: tuple[np.ndarray, np.ndarray],
    S: np.ndarray,
    mol: Optional[Mole] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates natural orbitals from a density matrix in a basis of spin-unrestricted MOs.

    Args:
        rdm1mo:     one-particle reduced density matrix in MO basis
        mo_coeff:   molecular orbital coefficients
        S:          atomic orbital overlap matrix
        mol:        Instance of pyscf.gto.Mole. Note that if mol.symmetry is enabled, the natural
                    orbitals will be symmetry adapted.

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1a, rdm1b = rdm1mo
    mos_a, mos_b = mo_coeff

    # Transform everything into the spin-up orbital basis.
    Sab = multi_dot([mos_a.T, S, mos_b])
    rdm1 = rdm1a + multi_dot([Sab, rdm1b, Sab.T])

    # Diagonalize the total density in alpha orbital basis.
    return restricted_natural_orbitals(rdm1, mos_a, mol)


def natural_spin_orbitals(
    rdm1mo: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    mo_coeff: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates natural spin orbitals from a density matrix in a basis of spin-unrestricted MOs.

    Args:
        rdm1mo:     one-particle reduced density matrix in MO basis, alpha and beta parts
        mo_coeff:   molecular orbital coefficients for alpha and beta orbitals

    Returns:
        Tuple with (natural spin occupation numbers (a, b), natural spin orbitals (a, b))

    Raises:
        ValueError: Invalid input.
    """
    # Sanity checking of the MO coefficients.
    mo_coeff = np.array(mo_coeff)
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2:
        raise ValueError("mo_coeff must consist of two N(AO) x N(MO) matrices.")

    # number of atomic basis functions
    nao = mo_coeff.shape[1]
    # number of MOs
    nmo = mo_coeff.shape[2]

    # Sanitization of the 1-RDM.
    rdm1mo = np.array(rdm1mo)
    if rdm1mo.shape != (2, nmo, nmo):
        raise ValueError("rdm1mo must consist of two N(MO) x N(MO) matrices")

    # Natural spin occupation numbers: alpha and beta.
    nsocc = np.zeros((2, nmo))
    # Natural spin orbital coefficients: alpha and beta.
    nsorb = np.zeros((2, nao, nmo))

    # Diagonalize the density matrix and order the natural spin orbitals by descending eigenvalues.
    for s in (0, 1):
        eigval, eigvec = eigh(rdm1mo[s])
        nsocc[s] = np.flip(eigval)
        nsorb[s] = np.dot(mo_coeff[s], np.fliplr(eigvec))

    return nsocc, nsorb


def ao_natural_orbitals(
    rdm1ao: np.ndarray, S: np.ndarray, Sthresh: float = DEFAULT_STHRESH
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates natural orbitals for a density matrix in AO basis.

    Args:
        rdm1ao:     one-particle reduced density matrix in AO basis
        S:          overlap matrix
        Sthresh:    eigenvalue cutoff to remove linear dependencies

    Returns:
        natural occupation numbers, natural orbitals
    """
    S12, Sminus12 = overlap_square_roots(S, Sthresh)

    # density matrix in the symmetrically orthogonalized basis
    rdm1orth = multi_dot([S12, rdm1ao, S12])
    eigval, eigvec = eigh(rdm1orth)

    # The diagonalization routine orders the eigenvalue from lowest to highest.
    # We want to have the occupied orbitals first and the unoccupied ones last.
    natocc = np.flip(eigval)
    natorb = np.dot(Sminus12, np.fliplr(eigvec))

    return natocc, natorb


def uhf_natural_orbitals(mf: UHFClass) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the natural orbitals for a converged UHF-type object.

    If symmetry is enabled in mf.mol, the natural orbitals will be symmetry-adapted.

    Args:
        mf:         UHF object with broken spin symmetry

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1mo = np.diag(mf.mo_occ[0]), np.diag(mf.mo_occ[1])
    S = mf.get_ovlp()
    return unrestricted_natural_orbitals(rdm1mo, mf.mo_coeff, S, mf.mol)


def mp2_natural_orbitals(pt: Union[RMP2Class, UMP2Class]) -> tuple[np.ndarray, np.ndarray]:
    """Calculates MP2 natural orbitals.

    Attempts to identify restricted/unrestricted basis automatically.
    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        pt:         An MP2 object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    mo_coeff = pt.mo_coeff
    if isinstance(mo_coeff, np.ndarray) and (mo_coeff.ndim == 2):
        return rmp2_natural_orbitals(pt)
    else:
        return ump2_natural_orbitals(pt)


def rmp2_natural_orbitals(pt: RMP2Class) -> tuple[np.ndarray, np.ndarray]:
    """Calculates restricted MP2 natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        pt:         An MP2 object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1 = pt.make_rdm1(ao_repr=False)
    mo_coeff = pt.mo_coeff
    return restricted_natural_orbitals(rdm1, mo_coeff, pt.mol)


def ump2_natural_orbitals(pt: UMP2Class) -> tuple[np.ndarray, np.ndarray]:
    """Calculates unrestricted MP2 natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        pt:         An MP2 object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1 = pt.make_rdm1(ao_repr=False)
    mo_coeff = pt.mo_coeff
    S = pt.mol.intor_symmetric("int1e_ovlp")
    return unrestricted_natural_orbitals(rdm1, mo_coeff, S, pt.mol)


def ccsd_natural_orbitals(cc: Union[RCCSDClass, UCCSDClass]) -> tuple[np.ndarray, np.ndarray]:
    """Calculates CCSD natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        cc:         A CCSD object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    return mp2_natural_orbitals(cc)


def rccsd_natural_orbitals(cc: RCCSDClass) -> tuple[np.ndarray, np.ndarray]:
    """Calculates spin-restricted CCSD natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        cc:         A CCSD object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    return rmp2_natural_orbitals(cc)


def uccsd_natural_orbitals(cc: UCCSDClass) -> tuple[np.ndarray, np.ndarray]:
    """Calculates spin-unrestricted CCSD natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        cc:         A CCSD object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    return ump2_natural_orbitals(cc)


def count_active_electrons(
    mo_occ: Union[Sequence[float], np.ndarray], mo_list: Union[Sequence[int], np.ndarray]
) -> int:
    """Counts the number of active electrons based on occupation numbers.

    It is assumed that the occupation numbers outside the active space round to 2 or 0.

    Args:
        mo_occ: list of orbital occupation numbers
        mo_list: list of orbitals in the active space

    Returns:
        number of active electrons

    Raises:
        Exception: various errors
    """
    # count total number of electrons
    # convert safely to an integer
    nelectrons_float = sum(mo_occ)
    electrons_total = iround(nelectrons_float)

    # Count the number of doubly occupied inactive orbitals
    # Check that the occupation numbers are not "unreasonable"
    number_docc = 0
    for i, occ in enumerate(mo_occ):
        if i not in mo_list:
            iocc = iround(occ)
            if iocc not in (0, 2):
                raise Exception("Occupation number outside mo_list does not round to 2 or 0.")
            elif iocc == 2:
                number_docc += 1

    # remove number of electrons in doubly occupied orbitals
    nel = electrons_total - 2 * number_docc
    return nel


def select_natural_occupations(
    natocc: np.ndarray,
    lower: float = 0.02,
    upper: float = 1.98,
    max_orb: Optional[int] = None,
    min_orb: Optional[int] = None,
) -> tuple[int, list[int]]:
    """Selects natural orbitals based on their eigenvalues between a lower and an upper boundary.

    If a maximal number of orbitals is provided, this function will truncate the orbital list.
    Orbitals with the highest occupation numbers will be removed first if the space is more than
    half occupied. Likewise, orbitals with the lowest occupation numbers will be removed first
    if the space is less than half occupied.

    In a similar way, providing a minimum number of orbitals will lead to an extension of the
    orbital list. The function will proceed such as to arrive closest to a half-occupied space.

    Args:
        natocc: list of natural occupation numbers
        lower: lower boundary for the natural occupation numbers
        upper: upper boundary for the natural occupation numbers
        max_orb: maximal number of orbitals
        min_orb: minimal number of orbitals

    Returns:
        number of electrons, list of MO indices (counting from zero)

    Raises:
        Exception: various errors
    """
    if natocc.ndim != 1:
        raise Exception("natocc must be a 1-D array")
    if lower > upper:
        raise Exception("Lower threshold must be smaller than the upper threshold.")
    if min_orb is not None and max_orb is not None:
        if min_orb > max_orb:
            raise Exception("Minimal number must be smaller than maximal number.")

    # order the natural occupation numbers, and store their original order
    sort_order = np.argsort(natocc)
    natocc_sorted = natocc[sort_order]
    N = len(natocc)

    # Partition the sorted list and count the active electrons.
    # Include occupations >= lower and <= upper, hence the 'side' argument.
    act_start = natocc_sorted.searchsorted(lower, side="left")
    act_end = natocc_sorted.searchsorted(upper, side="right")
    nel = count_active_electrons(natocc_sorted, np.arange(act_start, act_end))

    # If a minimum number of orbitals was provided, extend the orbital window if necessary.
    if min_orb is not None:
        while act_end - act_start < min_orb:
            # Can be extended both ways? Move closer to being half-occupied.
            if act_start > 0 and act_end < N:
                if nel < act_end - act_start:
                    act_end += 1
                else:
                    act_start -= 1
            # No virtuals left? Add occupied orbital.
            elif act_start == 0 and act_end < N:
                act_end += 1
            # No occupied orbitals left? Add virtual.
            elif act_start > 0 and act_end == N:
                act_start -= 1
            # Nothing left? We are done.
            elif act_start == 0 and act_end == N:
                break
            else:
                raise Exception("unknown error")
            nel = count_active_electrons(natocc_sorted, np.arange(act_start, act_end))

    # If a maximum number of orbitals was provided, make the orbital window smaller if necessary.
    if max_orb is not None:
        while act_end - act_start > max_orb:
            # Remove either occupied or virtual orbital to get closer to being half-occupied.
            if nel < act_end - act_start:
                act_start += 1
            else:
                act_end -= 1
            nel = count_active_electrons(natocc_sorted, np.arange(act_start, act_end))

    # Map back to the original orbital list and sort the indices.
    mo_list = np.sort(sort_order[np.arange(act_start, act_end)]).astype(int).tolist()
    return nel, mo_list


def extend_orbital_space(
    mo_list: Sequence[int],
    mo_occ: Sequence[float],
    Kmat: np.ndarray,
    docc_thresh: float = 1.5,
    unocc_thresh: float = 0.5,
) -> tuple[int, list[int]]:
    """Extend a set of active orbitals with correlation partners based on exchange integrals.

    Args:
        mo_list: initial list of active orbitals
        mo_occ: list of molecular orbital occupation numbers
        Kmat: matrix of exchange integrals, K_pq = (pq|pq)
        docc_thresh: occupation numbers above this threshold are considered doubly occupied
        unocc_thresh: occupation numbers below this threshold are considered unoccupied

    Returns:
        new number of electrons, extended list of orbitals

    Raises:
        Exception: input error
    """
    # Assumption: all orbitals that are not mostly doubly occupied or empty
    # are already in mo_list. Refuse to proceed if this assumption is violated.
    for i, occ in enumerate(mo_occ):
        if i not in mo_list:
            if (occ < docc_thresh) and (occ > unocc_thresh):
                raise Exception("Orbital with fractional occupation number outside mo_list.")

    mos_extended = list(mo_list)
    for i in mo_list:
        # Find the orbital index k which has the largest exchange integral with orbital i,
        # under the condition that the occupations are complementary.
        k = -1
        for k in np.flip(np.argsort(Kmat[i, :])):
            if k == i:
                continue
            elif (mo_occ[i] >= docc_thresh) and (mo_occ[k] < docc_thresh):
                break
            elif (mo_occ[i] <= unocc_thresh) and (mo_occ[k] > unocc_thresh):
                break
            elif (mo_occ[i] > unocc_thresh) and (mo_occ[i] < docc_thresh):
                break

        # orbital k has been determined as the partner orbital for i
        if k not in mos_extended:
            mos_extended.append(k)

    mos_extended.sort()
    nel = count_active_electrons(mo_occ, mos_extended)
    return nel, mos_extended
