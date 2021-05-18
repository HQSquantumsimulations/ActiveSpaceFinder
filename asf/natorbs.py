from typing import Tuple, Union, Sequence

import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import eigh

from pyscf.scf import UHF
from pyscf.mp.mp2 import RMP2 as RMP2Class
from pyscf.mp.ump2 import UMP2 as UMP2Class
from pyscf.cc.ccsd import CCSD as RCCSDClass
from pyscf.cc.uccsd import UCCSD as UCCSDClass

from asf.asf import overlapSquareRoots, iround


# Default eigenvalue cutoff to remove linear dependencies.
Sthresh_default = 1.0e-8


def restrictedNaturalOrbitals(rdm1mo: np.ndarray,
                              mo_coeff: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates natural orbitals in a basis of spin-restricted MOs.

    Args:
        rdm1mo: one-particle reduced density matrix in MO basis
        mo_coeff: molecular orbital coefficients

    Returns:
        natural occupation numbers, natural orbitals
    """
    eigval, eigvec = eigh(rdm1mo)

    # The diagonalization routine orders the eigenvalue from lowest to highest.
    # We want to have the occupied orbitals first and the unoccupied ones last.
    natocc = np.flip(eigval)
    natorb = np.dot(mo_coeff, np.fliplr(eigvec))

    return natocc, natorb


def unrestrictedNaturalOrbitals(rdm1mo: Tuple[np.ndarray, np.ndarray],
                                mo_coeff: Tuple[np.ndarray, np.ndarray],
                                S: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates natural orbitals in a basis of spin-unrestricted MOs.

    Args:
        rdm1mo: one-particle reduced density matrix in MO basis
        mo_coeff: molecular orbital coefficients
        S: atomic orbital overlap matrix

    Returns:
        natural occupation numbers, natural orbitals
    """
    rdm1a, rdm1b = rdm1mo
    mos_a, mos_b = mo_coeff

    # Transform everything into the spin-up orbital basis.
    Sab = multi_dot([mos_a.T, S, mos_b])
    rdm1 = rdm1a + multi_dot([Sab, rdm1b, Sab.T])

    # Diagonalize the total density in alpha orbital basis.
    return restrictedNaturalOrbitals(rdm1, mos_a)


def AOnaturalOrbitals(rdm1ao: np.ndarray,
                      S: np.ndarray,
                      Sthresh: float = Sthresh_default) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates natural orbitals for a density matrix in AO basis.

    Args:
        rdm1ao: one-particle reduced density matrix in AO basis
        S: overlap matrix
        Sthresh: eigenvalue cutoff to remove linear dependencies

    Returns:
        natural occupation numbers, natural orbitals
    """
    S12, Sminus12 = overlapSquareRoots(S, Sthresh)

    # density matrix in the symmetrically orthogonalized basis
    rdm1orth = multi_dot([S12, rdm1ao, S12])
    eigval, eigvec = eigh(rdm1orth)

    # The diagonalization routine orders the eigenvalue from lowest to highest.
    # We want to have the occupied orbitals first and the unoccupied ones last.
    natorb = np.flip(eigval)
    natocc = np.dot(Sminus12, np.fliplr(eigvec))

    return natorb, natocc


def UHFNaturalOrbitals(mf: UHF) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the natural orbitals for a converged UHF-type object.
    """
    rdm1mo = np.diag(mf.mo_occ[0]), np.diag(mf.mo_occ[1])
    S = mf.get_ovlp()
    return unrestrictedNaturalOrbitals(rdm1mo, mf.mo_coeff, S)


def MP2NaturalOrbitals(pt: Union[RMP2Class, UMP2Class]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates MP2 natural orbitals.
    Attempts to identify restricted/unrestricted basis automatically.
    """
    mo_coeff = pt.mo_coeff
    if isinstance(mo_coeff, np.ndarray) and (mo_coeff.ndim == 2):
        return RMP2NaturalOrbitals(pt)
    else:
        return UMP2NaturalOrbitals(pt)


def RMP2NaturalOrbitals(pt: RMP2Class) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates restricted MP2 natural orbitals.
    """
    rdm1 = pt.make_rdm1(ao_repr=False)
    mo_coeff = pt.mo_coeff
    return restrictedNaturalOrbitals(rdm1, mo_coeff)


def UMP2NaturalOrbitals(pt: UMP2Class) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates unrestricted MP2 natural orbitals.
    """
    rdm1 = pt.make_rdm1(ao_repr=False)
    mo_coeff = pt.mo_coeff
    S = pt.mol.intor_symmetric('int1e_ovlp')
    return unrestrictedNaturalOrbitals(rdm1, mo_coeff, S)


def CCSDNaturalOrbitals(cc: Union[RCCSDClass, UCCSDClass]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates CCSD natural orbitals.
    """
    return MP2NaturalOrbitals(cc)


def RCCSDNaturalOrbitals(cc: RCCSDClass) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates spin-restricted CCSD natural orbitals.
    """
    return RMP2NaturalOrbitals(cc)


def UCCSDNaturalOrbitals(cc: UCCSDClass) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates spin-unrestricted CCSD natural orbitals.
    """
    return UMP2NaturalOrbitals(cc)


def countActiveElectrons(mo_occ: Sequence[float], mo_list: Sequence[int]) -> int:
    """
    Counts the number of active electrons based on occupation numbers.
    It is assumed that the occupation numbers outside the active space round to 2 or 0.

    Args:
        mo_occ: list of orbital occupation numbers
        mo_list: list of orbitals in the active space

    Returns:
        number of active electrons
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
                             max_orb: int = None) -> Tuple[int, np.ndarray]:
    """
    Selects natural orbitals based on their eigenvalues between a lower and an upper boundary.

    Args:
        natocc: list of natural occupation numbers
        lower: lower boundary for the natural occupation numbers
        upper: upper boundary for the natural occupation numbers
        max_orb: maximal number of orbitals

    Returns:
        number of electrons, list of MO indices (counting from zero)
    """
    assert(natocc.ndim == 1)
    assert(lower < upper)

    # identify natural occupation numbers within the boundaries
    mo_list = []
    for i, n in enumerate(natocc):
        if (n >= lower) and (n <= upper):
            mo_list.append(i)

    # If the maximal number of orbitals exceeds the threshold,
    # keep removing orbitals such that the active space is approximately half occupied.
    if max_orb is not None:
        while len(mo_list) > max_orb:
            nel = countActiveElectrons(natocc, mo_list)
            if nel > len(mo_list):
                max_ind = np.argmax(natocc[mo_list])
                del mo_list[max_ind]
            else:
                min_ind = np.argmin(natocc[mo_list])
                del mo_list[min_ind]

    # count the electrons in the selected orbitals
    nel = countActiveElectrons(natocc, mo_list)

    return nel, np.array(mo_list)
