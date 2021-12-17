# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Core functions to calculate entropies and select orbitals."""

from typing import Tuple, List, Union
from abc import ABC, abstractmethod

from math import sqrt, log
import numpy as np
from numpy.linalg import eigh, multi_dot

from pyscf.gto import Mole


# Default parameters for the orbital selection function.
entropy_zero_cutoff = 1.0e-12
default_entropy_threshold = -0.1 * log(0.25)
default_plateau_threshold = 0.1


class ASFBase(ABC):
    """
    Abstract base class for correlated methods to perform orbital selection.

    The general idea is to select orbitals based on criteria including single-site entropies,
    which are computed from an approximate correlated method, for example DMRG. Implementations
    for specific quantum chemical methods should be derived from this base class. A user can
    perform automatic active space selection by following the steps below:

    # Initialize the object. ASFMethod is a subclass of ASFBase, implementing a specific quantum
    # chemical method. An initial orbital subspace to perform the calculation may be specified.
    sf = ASFMethod(...)

    # The correlated calculation (e.g. DMRG) is performed, storing information for the subsequent
    # orbital selection in this object.
    sf.calculate(...)

    # Finally, an active space is selected as a subset of the initially provided orbital space.
    # The method returns the number of selected active electrons and a list of active MOs.
    nel, mo_list = sf.selectOrbitals(...)
    """

    def __init__(self,
                 mol: Mole,
                 mo_coeff: np.ndarray,
                 nel: int = None,
                 norb: int = None,
                 mo_list: Union[List[int], np.ndarray] = None) -> None:
        """
        Setting the initial parameters for the calculation.

        A molecule object and the MO coefficient matrix must always be provided. Aside from that,
        the initializer is intended to be invoked one of the following ways:

        1) __init__(self, mol, mo_coeff)

            The correlated calculation will be performed using all orbitals. Active orbitals can
            be selected from the entire MO space.

        2) __init__(self, mol, mo_coeff, nel, norb)

            An initial active space of nel electrons in norb orbitals is specified by the user.
            The correlated calculation will be performed in this initial active space. Therefore,
            the final active space will be selected from the orbitals in the initial active space.

            The columns in mo_coeff should follow the convention of PySCF for CAS calculations.
            Practically, this implies:
            - The leftmost columns contain the inactive ("core") orbitals that are always doubly
              occupied.
            - The norb columns in the middle contain the initial subspace that the final active
              space will be selected from.
            - The rightmost columns contain the remaining orbitals that are always empty.

        3) __init__(self, mol, mo_coeff, nel=nel, mo_list=mo_list)

            As in option 2), but the user provides a list of orbitals for the initial orbital
            space. mo_list is specified analogously to pyscf.mcscf.sort_mo(..., base=0), so that
            counting starts from zero, not from one! The columns of mo_coeff should be ordered
            along the conventions of PySCF for CAS calculations:
            - mo_list specifies the orbitals to be included in the initial orbital space, which
              the final active space will be selected from.
            - The leftmost columns that are not in mo_list contain inactive ("core") orbitals that
              are always doubly occupied.
            - The remaining columns contain orbitals which are always empty.

        Args:
            mol:                Molecule object
            mo_coeff:           Spin-restricted molecular orbital coefficients
                                (AOs in rows, MOs in columns)
            nel:                Number of electrons in the initial orbital space
            norb:               Number of orbitals in the initial space
            mo_list:            List of orbitals for the initial space. Note that this follows the
                                convention of pyscf.mcscf.sort_mo(..., base=0): counting starts
                                from zero, not from one.

        Raises:
            Exception:          input error
        """
        self.mol = mol
        if mo_coeff.ndim != 2:
            raise Exception('mo_coeff must by a 2-D array.')
        self.mo_coeff = mo_coeff

        # option 1: calculation with all orbitals
        if nel is None and norb is None and mo_list is None:
            self.nel = mol.nelectron
            self.norb = self.mo_coeff.shape[1]
            self.ncore = calc_ncore(self.mol, self.nel)
            self.mo_list = np.arange(self.norb)
        # option 2: calculation with an (nel, norb) initial orbital space
        elif nel is not None and norb is not None and mo_list is None:
            self.nel = nel
            self.norb = norb
            self.ncore = calc_ncore(self.mol, nel)
            self.mo_list = np.arange(self.ncore, self.ncore + norb)
        # option 3: initial space with nel electrons and orbitals in mo_list
        elif nel is not None and mo_list is not None:
            # Permit the user to provide a value for norb, as long as it makes sense.
            if (norb is not None) and (norb != len(mo_list)):
                raise Exception('norb and mo_list are inconsistent.')
            self.nel = nel
            self.norb = len(mo_list)
            self.ncore = calc_ncore(self.mol, nel)
            self.mo_list = np.array(mo_list)
            if self.mo_list.ndim != 1:
                raise Exception('mo_list must be a 1-D array or a list')
        else:
            raise Exception('Incompatible combination of options nel, norb, mo_list.')

    @abstractmethod
    def calculate(self, *args, **kwargs) -> None:
        """
        Perform a correlated calculation, which generates the information used to select orbitals.

        Args:
            *args: placeholder for positional arguments
            **kwargs: placeholder for keyword arguments
        """
        raise NotImplementedError

    @abstractmethod
    def getOrbDens(self, root: int = 0) -> np.ndarray:
        """
        Calculate the one-orbital density.

        Args:
            root: electronic state to obtain the density for

        Returns:
            The one-orbital density as a N x 4 matrix for N orbitals.
        """
        raise NotImplementedError

    def calcEntropy(self,
                    orbdens: np.ndarray = None,
                    root: int = 0) -> np.ndarray:
        """
        Calculates the one-orbital entropy based on a one-orbital density.

        Args:
            orbdens: Orbital density to calculate the entropy with.
            root: electronic state to obtain the density for

        Returns:
            Vector of single-site entropies for the orbitals.
        """
        if orbdens is None:
            orbdens = self.getOrbDens(root=root)
        return calcS1(orbdens)

    def selectOrbitals(self,
                       root: int = 0,
                       threshold: float = default_entropy_threshold,
                       plateau_threshold: float = default_plateau_threshold,
                       verbose: bool = True) -> Tuple[int, List[int]]:
        """
        Performs the orbital selection.

        Args:
            root:               Electronic state to perform the orbital selection for.
            threshold:          Threshold for the single-site entropies
            plateau_threshold:  Threshold to identify plateaus
            verbose:            Print information or not

        Returns:
            number of electrons
            list of selected orbitals
        """
        orbdens = self.getOrbDens(root=root)
        entropy = self.calcEntropy(orbdens)
        nel, selected_mos = selectOrbitals(orbdens, threshold=threshold,
                                           plateau_threshold=plateau_threshold)

        if verbose:
            print(' MO    w( )    w(\u2191)    w(\u2193)    w(\u21c5)       S  sel')

            for i in self.mo_list:
                if i in selected_mos:
                    status = 'a'
                else:
                    status = ' '

                print('{0:3d}  {1:6.3f}  {2:6.3f}  {3:6.3f}  {4:6.3f}  {5:6.3f}  {6:>3s}'.
                      format(i, orbdens[i, 0], orbdens[i, 1], orbdens[i, 2], orbdens[i, 3],
                             entropy[i], status))

        return nel, selected_mos


def calc_ncore(mol: Mole, nel_act: int) -> int:
    """
    Calculates the number of inactive / core orbitals if the number of active electrons is known.

    Args:
        mol:        molecular information
        nel_act:    number of active electrons

    Returns:
        the number of core orbitals

    Raises:
        ValueError: error
    """
    ncoreelec = mol.nelectron - nel_act
    if ncoreelec % 2 != 0:
        raise ValueError('Obtained an odd number of core electrons.')
    return ncoreelec // 2


def inactive_orbital_lists(ncore: int, nmo: int, active_list: Union[List[int], np.ndarray]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Obtain lists of inactive core and virtual orbitals from an active space specification.

    Args:
        ncore:          number of core orbitals
        nmo:            total number of orbitals
        active_list:    list of active orbitals

    Returns:
        list of core orbitals, list of virtual orbitals

    Raises:
        ValueError: erratic input encountered
    """
    # Get all inactive orbitals as those that are not in the active space.
    inactive_orbitals = np.setdiff1d(np.arange(nmo), active_list)
    # sanity check
    if len(inactive_orbitals) + len(active_list) != nmo:
        raise ValueError('Active space definition is inconsistent with the number of MOs.')

    # partition inactive orbitals into core and virtuals
    core_orbitals = inactive_orbitals[:ncore]
    virtual_orbitals = inactive_orbitals[ncore:]
    return core_orbitals, virtual_orbitals


def iround(x: float) -> int:
    """
    A safe variant of rounding that always returns an int.

    Python's round can return float, e.g. if its argument is a numpy.float.

    Args:
        x:                      Floating point number to be rounded

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
        orbdens:                one-orbital densities in an N x 4 array
        zero_cutoff:            Threshold to determine values that are effectively zero

    Returns:
        single-site entropies

    Raises:
        Exception: various errors
    """
    if not ((orbdens.ndim == 2) and (orbdens.shape[1] == 4)):
        raise Exception('Orbital Density has wrong dimensions.')
    if not(zero_cutoff >= 0.0):
        raise Exception('Zero cut off should be larger than 0.')

    # Locations where the one-orbital density is positive
    where_positive = (orbdens > zero_cutoff)
    # Locations where the one-orbital density is zero (within the tolerance)
    where_zero = np.logical_and(orbdens <= zero_cutoff, orbdens >= -zero_cutoff)
    # Ensure there are no negative entries
    if not (np.all(orbdens >= -zero_cutoff)):
        raise Exception('Negative values in orb density.')

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
        orbdens:                one-orbital density with
                                columns representing empty, spin-up, spin-down, doubly occupied
        threshold:              Normal threshold for the entropy (unless there is a plateau)
        plateau_threshold:      Relative threshold to identify 'plateaus'
        zero_cutoff:            Threshold to neglect zeroes when calculating the entropy

    Returns:
        number of electrons
        list of orbitals

    Raises:
        Exception: various errors
    """
    if not ((orbdens.ndim == 2) and (orbdens.shape[1] == 4)):
        raise Exception('Wrong orbital density dimensions.')

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
