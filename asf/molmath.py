# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Reusable functions involving molecular descriptors such as orbitals and densities."""

from math import sqrt

import numpy as np
from numpy.linalg import eigh, multi_dot


def overlap_square_roots(S: np.ndarray, Sthresh: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculates square root and inverse square root of a matrix S.

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


def corresponding_orbitals(
    S12: np.ndarray, mo_coeff1: np.ndarray, mo_coeff2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates a corresponding orbital transformation between different MO sets.

    The original idea dates back to Amos and Hall (https://doi.org/10.1098/rspa.1961.0175).
    See also the work of Neese (https://doi.org/10.1016/j.jpcs.2003.11.015).

    Args:
        S12: overlap matrix between MOs 1 and MOs 2, can be rectangular
        mo_coeff1: first set of MO coefficients
        mo_coeff2: second set of MO coefficients

    Returns:
        corresponding orbital set 1, corresponding orbital set 2, singular values
    """
    u, sigma, vh = np.linalg.svd(S12)

    # The corresponding orbitals.
    co_coeff1 = np.dot(mo_coeff1, u)
    co_coeff2 = np.dot(mo_coeff2, vh.T)

    return co_coeff1, co_coeff2, sigma
