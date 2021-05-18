from typing import Tuple, List

from math import sqrt, log
import numpy as np
from numpy.linalg import eigh, multi_dot


# Default parameters for the orbital selection function.
entropy_zero_cutoff = 1.0e-12
default_entropy_threshold = -0.1 * log(0.25)
default_plateau_threshold = 0.1


def iround(x: float) -> int:
    """
    A safe variant of rounding that always returns an int.
    Python's round can return float, e.g. if its argument is a numpy.float.

    Args:
        x: floating point number to be rounded

    Returns:
        rounded number
    """
    if x >= 0.0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)


def calcS1(orbdens: np.ndarray,
           zero_cutoff: float = entropy_zero_cutoff) -> np.ndarray:
    """
    Calculates the single-site entropies from the one-orbital density.

    Args:
        orbdens: one-orbital densities in an N x 4 array
        zero_cutoff: threshold to determine values that are effectively zero

    Returns:
        single-site entropies
    """
    assert((orbdens.ndim == 2) and (orbdens.shape[1] == 4))
    assert(zero_cutoff >= 0.0)

    # Locations where the one-orbital density is positive
    where_positive = (orbdens > zero_cutoff)
    # Locations where the one-orbital density is zero (within the tolerance)
    where_zero = np.logical_and(orbdens <= zero_cutoff, orbdens >= -zero_cutoff)
    # Ensure there are no negative entries
    assert(np.all(orbdens >= -zero_cutoff))

    # Now calculate the actual entropies.
    WlogW = orbdens * np.log(orbdens, where=where_positive)
    WlogW[where_zero] = 0.0
    s1 = -np.sum(WlogW, axis=1)
    return s1


def selectOrbitals(orbdens: np.ndarray,
                   threshold: float = default_entropy_threshold,
                   plateau_threshold: float = default_plateau_threshold,
                   zero_cutoff: float = entropy_zero_cutoff) -> \
        Tuple[int, List[int]]:
    """
    Select orbitals with entropies above a given threshold.
    In addition, the list of chosen orbitals is extended
    - until the number of unpaired electrons is correct,
    - until the separation between entropies is sufficiently large ('plateaus'),
    - and with orbitals which are predominantly singly occupied.

    Args:
        orbdens: one-orbital density with
                 columns representing empty, spin-up, spin-down, doubly occupied
        threshold: normal threshold for the entropy (unless there is a plateau)
        plateau_threshold: relative threshold to identify 'plateaus'
        zero_cutoff: threshold to neglect zeroes when calculating the entropy

    Returns:
        number of electrons, list of orbitals
    """
    assert((orbdens.ndim == 2) and (orbdens.shape[1] == 4))

    Norb = orbdens.shape[0]
    entropies = calcS1(orbdens, zero_cutoff)

    # find orbital indices that sort the entropy in descending order
    entropy_sort = np.flip(np.argsort(entropies))

    # orbitals above the entropy threshold are always included
    selected = []
    for i in entropy_sort:
        if entropies[i] >= threshold:
            selected.append(i)
        else:
            break

    # identify orbitals that are part of a trailing 'plateau'
    if selected:
        for i in entropy_sort[len(selected):]:
            if entropies[i] >= entropies[selected[-1]] * (1.0 - plateau_threshold):
                selected.append(i)
            else:
                break

    # identify orbitals that are predominantly singly occupied (spin up or spin down)
    for i in range(Norb):
        if orbdens[i, 1] + orbdens[i, 2] > orbdens[i, 0] + orbdens[i, 3]:
            if i not in selected:
                selected.append(i)

    selected.sort()

    # count the total number of electrons
    orbital_electrons = orbdens[:, 1] + orbdens[:, 2] + 2.0 * orbdens[:, 3]
    electrons_total = sum(orbital_electrons[selected])

    return iround(electrons_total), selected


def overlapSquareRoots(S: np.ndarray,
                       Sthresh: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates square root and inverse square root of a matrix S.

    Args:
        S: (overlap) matrix
        Sthresh: eigenvalue cutoff to remove linear dependencies

    Returns:
        S^1/2, S^-1/2
    """
    eigval, eigvec = eigh(S)

    # Diagonal matrices containing the square roots or the inverse square roots
    # of the eigenvalues of S.
    A12 = np.zeros(S.shape)
    Am12 = np.zeros(S.shape)
    for i, v in enumerate(eigval):
        if v > Sthresh:
            A12[i, i] = sqrt(v)
            Am12[i, i] = 1.0 / sqrt(v)

    # square root of S
    S12 = multi_dot([eigvec, A12, eigvec.T])

    # inverse square root of S
    Sminus12 = multi_dot([eigvec, Am12, eigvec.T])

    return S12, Sminus12
