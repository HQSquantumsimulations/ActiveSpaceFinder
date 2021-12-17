# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Functions to calculate different types of natural orbitals."""

from typing import Tuple, Union, Sequence

import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import eigh

from pyscf.scf import UHF
from pyscf.mp.mp2 import RMP2 as RMP2Class
from pyscf.mp.ump2 import UMP2 as UMP2Class
from pyscf.cc.ccsd import CCSD as RCCSDClass
from pyscf.cc.uccsd import UCCSD as UCCSDClass
from pyscf.gto import Mole
from pyscf import symm

from asf.asf import overlapSquareRoots, iround

# Default eigenvalue cutoff to remove linear dependencies.
Sthresh_default = 1.0e-8


def restrictedNaturalOrbitals(rdm1mo: np.ndarray,
                              mo_coeff: np.ndarray,
                              mol: Mole = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates natural orbitals in a basis of spin-restricted MOs.

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


def unrestrictedNaturalOrbitals(rdm1mo: Tuple[np.ndarray, np.ndarray],
                                mo_coeff: Tuple[np.ndarray, np.ndarray],
                                S: np.ndarray,
                                mol: Mole = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates natural orbitals in a basis of spin-unrestricted MOs.

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
    return restrictedNaturalOrbitals(rdm1, mos_a, mol)


def AOnaturalOrbitals(rdm1ao: np.ndarray,
                      S: np.ndarray,
                      Sthresh: float = Sthresh_default) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates natural orbitals for a density matrix in AO basis.

    Args:
        rdm1ao:     one-particle reduced density matrix in AO basis
        S:          overlap matrix
        Sthresh:    eigenvalue cutoff to remove linear dependencies

    Returns:
        natural occupation numbers, natural orbitals
    """
    S12, Sminus12 = overlapSquareRoots(S, Sthresh)

    # density matrix in the symmetrically orthogonalized basis
    rdm1orth = multi_dot([S12, rdm1ao, S12])
    eigval, eigvec = eigh(rdm1orth)

    # The diagonalization routine orders the eigenvalue from lowest to highest.
    # We want to have the occupied orbitals first and the unoccupied ones last.
    natocc = np.flip(eigval)
    natorb = np.dot(Sminus12, np.fliplr(eigvec))

    return natocc, natorb


def UHFNaturalOrbitals(mf: UHF) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the natural orbitals for a converged UHF-type object.

    If symmetry is enabled in mf.mol, the natural orbitals will be symmetry-adapted.

    Args:
        mf:         UHF object with broken spin symmetry

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1mo = np.diag(mf.mo_occ[0]), np.diag(mf.mo_occ[1])
    S = mf.get_ovlp()
    return unrestrictedNaturalOrbitals(rdm1mo, mf.mo_coeff, S, mf.mol)


def MP2NaturalOrbitals(pt: Union[RMP2Class, UMP2Class]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates MP2 natural orbitals.

    Attempts to identify restricted/unrestricted basis automatically.
    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        pt:         An MP2 object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    mo_coeff = pt.mo_coeff
    if isinstance(mo_coeff, np.ndarray) and (mo_coeff.ndim == 2):
        return RMP2NaturalOrbitals(pt)
    else:
        return UMP2NaturalOrbitals(pt)


def RMP2NaturalOrbitals(pt: RMP2Class) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates restricted MP2 natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        pt:         An MP2 object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1 = pt.make_rdm1(ao_repr=False)
    mo_coeff = pt.mo_coeff
    return restrictedNaturalOrbitals(rdm1, mo_coeff, pt.mol)


def UMP2NaturalOrbitals(pt: UMP2Class) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates unrestricted MP2 natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        pt:         An MP2 object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1 = pt.make_rdm1(ao_repr=False)
    mo_coeff = pt.mo_coeff
    S = pt.mol.intor_symmetric('int1e_ovlp')
    return unrestrictedNaturalOrbitals(rdm1, mo_coeff, S, pt.mol)


def CCSDNaturalOrbitals(cc: Union[RCCSDClass, UCCSDClass]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates CCSD natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        cc:         A CCSD object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    return MP2NaturalOrbitals(cc)


def RCCSDNaturalOrbitals(cc: RCCSDClass) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates spin-restricted CCSD natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        cc:         A CCSD object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    return RMP2NaturalOrbitals(cc)


def UCCSDNaturalOrbitals(cc: UCCSDClass) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates spin-unrestricted CCSD natural orbitals.

    If symmetry is enabled, the natural orbitals will be symmetry-adapted.

    Args:
        cc:         A CCSD object.

    Returns:
        natural occupation numbers, natural orbitals
    """
    return UMP2NaturalOrbitals(cc)


def countActiveElectrons(mo_occ: Union[Sequence[float], np.ndarray],
                         mo_list: Union[Sequence[int], np.ndarray]) -> int:
    """
    Counts the number of active electrons based on occupation numbers.

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
                raise Exception('Occupation number outside mo_list does not round to 2 or 0.')
            elif iocc == 2:
                number_docc += 1

    # remove number of electrons in doubly occupied orbitals
    nel = electrons_total - 2 * number_docc
    return nel


def selectNaturalOccupations(natocc: np.ndarray,
                             lower: float = 0.02,
                             upper: float = 1.98,
                             max_orb: int = None,
                             min_orb: int = None) -> Tuple[int, np.ndarray]:
    """
    Selects natural orbitals based on their eigenvalues between a lower and an upper boundary.

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
        raise Exception('natocc must be a 1-D array')
    if lower > upper:
        raise Exception('Lower threshold must be smaller than the upper threshold.')
    if min_orb is not None and max_orb is not None:
        if min_orb > max_orb:
            raise Exception('Minimal number must be smaller than maximal number.')

    # order the natural occupation numbers, and store their original order
    sort_order = np.argsort(natocc)
    natocc_sorted = natocc[sort_order]
    N = len(natocc)

    # Partition the sorted list and count the active electrons.
    # Include occupations >= lower and <= upper, hence the 'side' argument.
    act_start = natocc_sorted.searchsorted(lower, side='left')
    act_end = natocc_sorted.searchsorted(upper, side='right')
    nel = countActiveElectrons(natocc_sorted, np.arange(act_start, act_end))

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
                raise Exception('unknown error')
            nel = countActiveElectrons(natocc_sorted, np.arange(act_start, act_end))

    # If a maximum number of orbitals was provided, make the orbital window smaller if necessary.
    if max_orb is not None:
        while act_end - act_start > max_orb:
            # Remove either occupied or virtual orbital to get closer to being half-occupied.
            if nel < act_end - act_start:
                act_start += 1
            else:
                act_end -= 1
            nel = countActiveElectrons(natocc_sorted, np.arange(act_start, act_end))

    # Map back to the original orbital list and sort the indices.
    mo_list = np.sort(sort_order[np.arange(act_start, act_end)])
    return nel, mo_list
