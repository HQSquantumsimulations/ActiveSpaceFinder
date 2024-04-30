# Copyright © 2020-2022 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Functions to calculate pair information between orbitals."""

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from math import isclose
from operator import attrgetter
from typing import Iterable, Optional, Sequence, Union, cast

import numpy as np
from pyscf.gto import Mole

from .utility import calculate_entropy_s1, iround


def cumulant2b(rdm1a: np.ndarray, rdm1b: np.ndarray, rdm2: np.ndarray) -> np.ndarray:
    """Calculates the two-body cumulant.

    It is computed as the difference between the two-body density matrix,
    and the antisymmetrized product of the one-body density matrices.

    Args:
        rdm1a: alpha component of the 1-RDM
        rdm1b: beta component of the 1-RDM
        rdm2: spin-traced(!) 2-RDM

    Returns:
        spin-traced cumulant <p^+ r^+ s q> - <p^+ q> <r^+ s> + <p^+ s> <r^+ q>

    Raises:
        Exception: input error
    """
    N = rdm2.shape[0]
    if rdm1a.shape != (N, N):
        raise Exception("rdm1a with wrong shape")
    if rdm1b.shape != (N, N):
        raise Exception("rdm1b with wrong shape")
    if rdm2.shape != (N, N, N, N):
        raise Exception("rdm2 with wrong shape")
    rdm1 = rdm1a + rdm1b

    # pyscf convention: rdm2[p, q, r, s] = <p^+ r^+ s q> + remaining spin traced terms
    #
    # Coulomb-like contribution: <p^+ q> <r^+ s> + ...
    J = np.einsum("pq,rs->pqrs", rdm1, rdm1)

    # exchange-like contribution: <p^+ s> <r^+ q> + ...
    K = np.einsum("ps,rq->pqrs", rdm1a, rdm1a) + np.einsum("ps,rq->pqrs", rdm1b, rdm1b)

    # the cumulant
    C = rdm2 - J + K
    return C


def cumulant2bs(
    rdm1a: np.ndarray,
    rdm1b: np.ndarray,
    rdm2aa: np.ndarray,
    rdm2ab: np.ndarray,
    rdm2bb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the spin components of the two-body cumulant.

    It is computed as the difference between the respective two-body density matrix,
    and the antisymmetrized product of the respective one-body density matrices.

    Args:
        rdm1a:  alpha component of the 1-RDM
        rdm1b:  beta component of the 1-RDM
        rdm2aa: pure alpha component of the 2-RDM, <p^+ r^+ s q>
        rdm2ab: mixed spin component of the 2-RDM, <p^+ r^+ s q> with p, q alpha, r, s beta
        rdm2bb: pure beta component of the 2-RDM, <p^+ r^+ s q>

    Returns:
        Tuple with (aa, ab, bb) parts of the cumulant.

    Raises:
        ValueError: input error
    """
    Na = rdm1a.shape[0]
    Nb = rdm1b.shape[0]
    if rdm1a.shape != (Na, Na):
        raise ValueError("'rdm1a' not a square matrix")
    if rdm1b.shape != (Nb, Nb):
        raise ValueError("'rdm1b' not a square matrix")
    if rdm2aa.shape != (Na, Na, Na, Na):
        raise ValueError("'rdm2aa' dimensions not consistent with rdm1a")
    if rdm2ab.shape != (Na, Na, Nb, Nb):
        raise ValueError("'rdm2ab' dimensions not consistent with rdm1a and rdm1b")
    if rdm2bb.shape != (Nb, Nb, Nb, Nb):
        raise ValueError("'rdm2bb' dimensions not consistent with rdm1b")

    # Calculate the pure alpha contribution to the cumulant.
    Jaa = np.einsum("pq,rs->pqrs", rdm1a, rdm1a)
    Kaa = np.einsum("ps,rq->pqrs", rdm1a, rdm1a)
    Caa = rdm2aa - Jaa + Kaa
    del Jaa, Kaa

    # Mixed-spin contribution to the cumulant: no "exchange-like" part.
    Jab = np.einsum("pq,rs->pqrs", rdm1a, rdm1b)
    Cab = rdm2ab - Jab
    del Jab

    # Calculate the pure beta contribution to the cumulant.
    Jbb = np.einsum("pq,rs->pqrs", rdm1b, rdm1b)
    Kbb = np.einsum("ps,rq->pqrs", rdm1b, rdm1b)
    Cbb = rdm2bb - Jbb + Kbb
    del Jbb, Kbb

    return (Caa, Cab, Cbb)


def transform_tensor(x: np.ndarray, A: Union[np.ndarray, tuple[np.ndarray, ...]]) -> np.ndarray:
    """Performs a basis transformation of a tensor.

    Args:
        x:  The tensor to be transformed.
        A:  The transformation matrix or a tuple of transformation matrices:
            rows in original basis, columns in new basis.
            If a single matrix is supplied, all dimensions of x are transformed using this matrix.
            If a tuple of matrices is supplied, matrix n will be used to transform dimension n.

    Returns:
        Transformed tensor y: y_pqr sum_ijk... x_ijk... A1_ip A2_jq A3_kr ...

    Raises:
        ValueError: Invalid argument values.
        TypeError:  Unsupported argument type.
    """
    ndim = x.ndim
    # If a tuple was provided, ensure its length matches the number of tensor dimensions.
    if isinstance(A, tuple):
        if len(A) != ndim:
            raise ValueError(
                "Number of transformation matrices not matching the tensor dimension."
            )
    # If a single matrix was provided, create the appropriate tuple.
    elif isinstance(A, np.ndarray):
        A = (A,) * ndim
    else:
        raise TypeError("Unsupported type of argument 'A'.")

    # Ensure all transformation matrices indeed have dimension two.
    for An in A:
        if An.ndim != 2:
            raise ValueError("Transformation matrices must be arrays with dimension 2.")

    # In each iteration, the first index of the input x becomes the last index of the output x.
    # -> Cycle through the indices.
    for n in range(ndim):
        x = np.tensordot(x, A[n], axes=([0], [0]))

    return x


def transform_unrestricted_rdm12s(
    mo_orig: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    mo_new: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    S_or_mol: Union[np.ndarray, Mole],
    rdm1s: Optional[Union[tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
    rdm2s: Optional[Union[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> tuple[
    Optional[tuple[np.ndarray, np.ndarray]], Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]
]:
    """Transform full 1-RDMs and 2-RDMs between two spin-orbital basis sets.

    Args:
        mo_orig:    Original orbital basis for the density matrix. It can be specified in two ways:
                    1) Tuple with two matrices: separate alpha and beta spin orbitals.
                    2) Single matrix: identical alpha and beta orbitals.
        mo_new:     New orbital basis, to which the density matrix will be transformed. As mo_orig,
                    it can also be specified either as a single matrix or as two matrices.
        S_or_mol:   There are two options for this argument:
                    1) The overlap matrix of atomic basis functions.
                    2) Molecular data as a Mole object, which will be used to obtain the overlap.
        rdm1s:      One-particle spin density matrix (alpha, beta).
        rdm2s:      Two-particle spin density matrix (aaaa, aabb, bbbb) with a: alpha, b: beta.

    Returns:
        Tuple containing transformed (rdm1s, rdm2s). rdm1s and rdm2s are tuples, respectively, with
        indices (aa, bb) for rdm1s and (aaaa, aabb, bbbb) for rdm2s. If an original rdm1s or rdm2s
        was not provided, None instead of the respective transformed density will be returned.

    Raises:
        ValueError: Invalid input.
    """
    # Calculate overlap matrix, or assign alias S.
    if isinstance(S_or_mol, Mole):
        S = S_or_mol.intor_symmetric("int1e_ovlp")
    elif isinstance(S_or_mol, np.ndarray) and S_or_mol.ndim == 2:
        S = S_or_mol
    else:
        raise ValueError("'S_or_mol' must be an overlap matrix or a Mole instance.")

    # Convert tuples with spin orbitals (if provided) to 2 x N x N arrays.
    if isinstance(mo_orig, tuple):
        mo_orig = np.array(mo_orig)
    if isinstance(mo_new, tuple):
        mo_new = np.array(mo_new)

    # If single matrices with identical alpha and beta orbitals were provided, replicate the spin
    # components to obtain 2 x N x N arrays.
    if mo_orig.ndim == 2:
        mo_orig = np.array([mo_orig, mo_orig])
    if mo_new.ndim == 2:
        mo_new = np.array([mo_new, mo_new])

    # Orthogonal basis transformation matrices for alpha and beta orbitals.
    Ua = mo_orig[0].T @ S @ mo_new[0]
    Ub = mo_orig[1].T @ S @ mo_new[1]

    # Transform the 1-RDM if it was provided.
    rdm1s_transformed = None
    if rdm1s is not None:
        rdm1s_transformed = transform_tensor(rdm1s[0], Ua), transform_tensor(rdm1s[1], Ub)

    # Transform the 2-RDM if it was provided.
    rdm2s_transformed = None
    if rdm2s is not None:
        rdm2aa_t = transform_tensor(rdm2s[0], Ua)
        rdm2ab_t = transform_tensor(rdm2s[1], (Ua, Ua, Ub, Ub))
        rdm2bb_t = transform_tensor(rdm2s[2], Ub)
        rdm2s_transformed = rdm2aa_t, rdm2ab_t, rdm2bb_t

    # Return the transformed RDMs as a tuple.
    return rdm1s_transformed, rdm2s_transformed


def transform_unrestricted_rdm1s(
    mo_orig: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    mo_new: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    S_or_mol: Union[np.ndarray, Mole],
    rdm1s: Union[tuple[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform 1-RDM between two spin-orbital bases.

    Args:
        mo_orig:    Original orbital basis for the density matrix. It can be specified in two ways:
                    1) Tuple with two matrices: separate alpha and beta spin orbitals.
                    2) Single matrix: identical alpha and beta orbitals.
        mo_new:     New orbital basis, to which the density matrix will be transformed. As mo_orig,
                    it can also be specified either as a single matrix or as two matrices.
        S_or_mol:   There are two options for this argument:
                    1) The overlap matrix of atomic basis functions.
                    2) Molecular data as a Mole object, which will be used to obtain the overlap.
        rdm1s:      One-particle spin density matrix (alpha, beta).

    Returns:
        Transformed 1-RDM as a tuple with (alpha part, beta part).
    """
    rdm1s_transformed = transform_unrestricted_rdm12s(mo_orig, mo_new, S_or_mol, rdm1s=rdm1s)[0]
    return cast(tuple[np.ndarray, np.ndarray], rdm1s_transformed)


def transform_unrestricted_rdm2s(
    mo_orig: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    mo_new: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    S_or_mol: Union[np.ndarray, Mole],
    rdm2s: Union[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform 2-RDM between two spin-orbital bases.

    Args:
        mo_orig:    Original orbital basis for the density matrix. It can be specified in two ways:
                    1) Tuple with two matrices: separate alpha and beta spin orbitals.
                    2) Single matrix: identical alpha and beta orbitals.
        mo_new:     New orbital basis, to which the density matrix will be transformed. As mo_orig,
                    it can also be specified either as a single matrix or as two matrices.
        S_or_mol:   There are two options for this argument:
                    1) The overlap matrix of atomic basis functions.
                    2) Molecular data as a Mole object, which will be used to obtain the overlap.
        rdm2s:      Two-particle spin density matrix (aaaa, aabb, bbbb) with a: alpha, b: beta.

    Returns:
        Transformed 2-RDM as a tuple (aaaa, aabb, bbbb) with a: alpha, b: beta.
    """
    rdm2s_transformed = transform_unrestricted_rdm12s(mo_orig, mo_new, S_or_mol, rdm2s=rdm2s)[1]
    return cast(tuple[np.ndarray, np.ndarray, np.ndarray], rdm2s_transformed)


def somo_from_orbdens(orbdens: np.ndarray) -> list[int]:
    """Identify the most appropriate ROHF-like singly occupied molecular orbitals.

    The one-orbital density is used to obtain both the total number of electrons and the net number
    of unpaired electrons. Based on their occupation numbers, the most suitable <Na> - <Nb> MOs are
    identified.

    Args:
        orbdens: Expectation values for vacant, spin-up, spin-down and double occupations.

    Returns:
        List of indices for a number of SOMOs matching the net number of unpaired electrons.
    """
    # Number of orbitals.
    norb = orbdens.shape[0]

    if orbdens.shape != (norb, 4):
        raise ValueError("'orbdens' must be a matrix with dimensions: norb x 4.")

    # Number of electrons per orbital.
    nelec_per_mo = orbdens[:, 1] + orbdens[:, 2] + 2.0 * orbdens[:, 3]

    # Total number of electrons.
    nelec = iround(np.sum(nelec_per_mo))

    # Calculate the net number of unpaired electrons.
    nspin = iround(np.sum(orbdens[:, 1] - orbdens[:, 2]))

    if nspin < 0:
        raise ValueError("Negative number of unpaired electrons.")
    if nspin > nelec:
        raise ValueError("Number of unpaired electrons larger than total number of electrons.")
    if (nelec - nspin) % 2 != 0:
        raise ValueError(f"nelec={nelec}, nspin={nspin}: odd number of paired electrons.")

    # Number of "doubly occupied" orbitals.
    ndomo = (nelec - nspin) // 2

    # Vector of "singly occupied" orbitals.
    somo_list = np.flip(np.argsort(nelec_per_mo))[ndomo : ndomo + nspin]

    # Return sorted list of integers.
    return sorted(map(int, somo_list))


@dataclass
class MOListInfo:
    """Information on a molecular orbital list for an active space.

    Attributes:
        mo_list:       List of molecular orbital indices.
        nel:           Number of electrons.
        minimal:       Flag indicating if this is a minimal active space.
        pairinfo_sum:  Sum of pair information over the MOs in mo_list and their edges.
        max_increment: Maximum amount of pair information that can be added with any orbital
                       outside of the current active space.
        min_decrement: Minimum amount of pair information that can be removed with any removable
                       orbital (e.g. within the active space, but not part of the minimum space).
        min_entropy:   Minimum entropy among removable MOs.
        min_edge_sum:  Minimum sum over all edges (active and inactive) among removable MOs.
    """

    mo_list: list[int]
    nel: int
    minimal: bool
    pairinfo_sum: float
    max_increment: float
    min_decrement: float
    min_entropy: float
    min_edge_sum: float

    def copy_with_indices(self, mo_list: list[int]) -> "MOListInfo":
        """Create copy of instance with different set of MO indices.

        Args:
            mo_list: List of MO indices.

        Returns:
            MOListInfo with new mo_list.
        """
        return replace(self, mo_list=mo_list)


# Dictionary type to store multiple suggestions for active spaces.
ActiveSpacesDict = dict[
    tuple[int, int],  # Keys: Active space identifiers (electrons, orbitals).
    list[MOListInfo],  # Values: List of active space suggestions for (electrons, orbitals) spaces.
]


def map_spaces_to_full(all_spaces: ActiveSpacesDict, mo_list_full: list[int]) -> ActiveSpacesDict:
    """Copy an active space collection with MO indices mapped onto the full MO space.

    In some cases ActiveSpacesDict contains MOListInfo objects that refer to relative MO indices.
    Given MO indices in the full MO space, this function maps the relative MO indices onto the
    full space.

    Args:
        all_spaces:    active space collection.
        mo_list_full:  list of indices of initial MOs in the full MO space. The indices are
                       used to map the relative indices to the full MO space.

    Returns:
        active space collection with MO indices mapped to full space
    """
    return {
        nel_nmo: [
            mo_info.copy_with_indices([mo_list_full[i] for i in mo_info.mo_list])
            for mo_info in space_variants
        ]
        for nel_nmo, space_variants in all_spaces.items()
    }


def less_or_equal(val1: float, val2: float, rel_tol: float, abs_tol: float = 1e-8) -> bool:
    """Performs a less-than-or-equal comparison to within a numerical tolerance.

    Args:
        val1:       First (smaller) value.
        val2:       Second (larger) value.
        rel_tol:    Relative comparison tolerance as defined in math.isclose.
        abs_tol:    Absolute comparison tolerance as defined in math.isclose.

    Returns:
        True if val1 < val2 or val1 is close to val2 to within the tolerances, False otherwise.
    """
    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError("Comparison tolerances must be positive.")
    return val1 < val2 or isclose(val1, val2, abs_tol=abs_tol, rel_tol=rel_tol)


def group_active_spaces(spaces: list[MOListInfo]) -> ActiveSpacesDict:
    """Group active spaces with equal number of active electrons and orbitals.

    The active spaces are grouped in a dictionary where the keys are CAS tuples (N, M), i.e. N
    active electrons in M active orbitals, and the values are lists of MOListInfo objects sorted
    in descending order by the pair information sum.

    Args:
        spaces: List of active spaces as MOListInfo objects.

    Returns:
        Active spaces grouped by CAS tuple (N, M).
    """
    mapped: ActiveSpacesDict = {}
    for mo_info in spaces:
        cas_tuple = (mo_info.nel, len(mo_info.mo_list))
        if cas_tuple in mapped:
            mapped[cas_tuple].append(mo_info)
        else:
            mapped[cas_tuple] = [mo_info]

    for cas_tuple in mapped.keys():
        mapped[cas_tuple].sort(key=attrgetter("pairinfo_sum"), reverse=True)

    return mapped


class PairInfoAnalyzer:
    """Functionality for the analysis of pair information to determine active space suggestions."""

    def __init__(
        self,
        pairinfo: np.ndarray,
        orbdens: np.ndarray,
        spin: Optional[int] = None,
    ) -> None:
        """The constructor sets various data used by the methods of the class.

        Args:
            pairinfo:    N x N matrix with the pair information: large, positive is most favorable.
            orbdens:     One-orbital density as N x 4 matrix (unocc., spin-up, spin-down, d. occ.)
            spin:        The number of unpaired electrons. Determined from orbdens if not set.
        """
        # Checking and storing the one-orbital density.
        if orbdens.ndim != 2 or orbdens.shape[1] != 4:
            raise ValueError("'orbdens' must be an N x 4 matrix.")
        self.orbdens = orbdens

        # Number of orbitals defined by the orbital density and the pair information.
        self.norb = orbdens.shape[0]

        # Ensure pair information matrix has correct dimensions, and store it.
        if pairinfo.shape != (self.norb, self.norb):
            raise ValueError("'pairinfo' must be an N x N matrix.")
        self.pairinfo = pairinfo

        # Spin polarization per orbital: 2 * <Sz>.
        self.mo_spin = orbdens[:, 1] - orbdens[:, 2]
        self.spin = spin if spin is not None else iround(np.sum(self.mo_spin))

        # Expectation number of electrons in each MO.
        self.electrons_per_mo = orbdens[:, 1] + orbdens[:, 2] + 2.0 * orbdens[:, 3]

        # Buffer to store pair information for each active space computed.
        self.pairinfo_sum_buffer: dict[tuple[int, ...], float] = {}

    def pairinfo_sum(self, mo_list: Iterable[int]) -> float:
        """Calculate the pair information sum for the active space specified.

        Calculated sums are stored in this object. Results that have been calculated previously are
        retrieved from the buffer instead of recalculating them.

        Args:
            mo_list:    List of orbital indices for which to calculate the pair information sum.

        Returns:
            The pair information sum for the specified orbital subspace.
        """
        # mo_list is converted to a tuple to make it hashable.
        mo_tuple = tuple(sorted(mo_list))

        if mo_tuple in self.pairinfo_sum_buffer:
            # Retrieve previously computed result if available...
            result = self.pairinfo_sum_buffer[mo_tuple]

        else:
            # ... or calculate the result and store it.
            result = np.sum(self.pairinfo[mo_tuple, :][:, mo_tuple])
            self.pairinfo_sum_buffer[mo_tuple] = result

        return result

    def pairinfo_per_orbital(self) -> np.ndarray:
        """Calculate the contribution of each orbital to the total pair information sum.

        Returns:
            Array containing the pair information sum per orbital.
        """
        return np.sum(self.pairinfo, axis=1)

    def minimal_space(self, mo_list_full: Optional[list[int]] = None) -> list[int]:
        """Construct a minimal active space based on the spin of orbitals.

        The active space is chosen such that it is guaranteed to round to the (integer) number of
        unpaired electrons from the orbital density; provided that the full set of MOs in orbdens
        adds to an integer number of unpaired electrons. This is achieved by ensuring that the spin
        of all orbitals outside the active space sums to no more than 0.5 for positive spin and no
        less than -0.5 for negative spin.

        Args:
            map_to_full_space:  Whether to map active MO indices onto the full MO space.
                                Note that 'initial_mos' is required for the mapping.
            mo_list_full:       List of indices of initial MOs in the full MO space. The indices
                                are used to map the relative indices to the full MO space.

        Returns:
            List of MO indices for a minimal active space.
        """
        # Vector with positive spin polarizations (negative values set to zero).
        mo_spin_plus = self.mo_spin.copy()
        mo_spin_plus[mo_spin_plus < 0.0] = 0.0

        # Vector with negative spin polarizations (positive values set to zero).
        mo_spin_minus = self.mo_spin.copy()
        mo_spin_minus[mo_spin_minus > 0.0] = 0.0

        # Indices that sort positively spin-polarized MOs in ascending order, and negatively spin-
        # polarized MOs in descending order (increasing magnitude in each case).
        argsort_plus = np.argsort(mo_spin_plus)
        argsort_minus = np.flip(np.argsort(mo_spin_minus))

        # Calculate the cumulated sums of positive and negative spins separately, each in the order
        # of increasing magnitude.
        cumsum_plus = np.cumsum(mo_spin_plus[argsort_plus])
        cumsum_minus = np.cumsum(mo_spin_minus[argsort_minus])

        # MOs with a cumulated positive spin polarization below 0.5, or a cumulated negative spin
        # polarization above 0.5 are not in the minimal space; the remaining MOs are in the minimal
        # space.
        mo_list: list[int] = (
            np.union1d(argsort_plus[cumsum_plus >= 0.5], argsort_minus[cumsum_minus <= -0.5])
            .astype(int)
            .tolist()
        )

        return mo_list if mo_list_full is None else [mo_list_full[i] for i in mo_list]

    def generate_candidate_spaces(
        self,
        minimal_space: Optional[list[int]] = None,
        mo_list_full: Optional[list[int]] = None,
    ) -> ActiveSpacesDict:
        """Construct suggestions for active spaces based on orbital pair information.

        Starting from a minimal (or empty) active space, active spaces are grown by extending them
        with one or two orbitals at a time. Multiple lists of orbitals are created for each number
        of active electrons in active orbitals, (nel, norb).

        A heuristic based on pair information is used to truncate the sets of spaces: for a given
        (N, M) combination, all MO lists must have a pair information sum exceeding a threshold.
        This threshold is the maximum pair information sum for an (nel, norb) space, minus the
        associated decrement.

        Args:
            minimal_space: Minimal active space with orbitals that must always be included. If None
                           is specified, a minimal space is constructed from the orbital density to
                           include MOs that substantially affect the spin.
            mo_list_full:  List of indices of initial MOs in the full MO space. The indices are
                           used to map the relative indices to the full MO space.

        Returns:
            Dictionary with active space suggestions.
            Keys:   (nel, norb) where nel and norb are the numbers of active electrons and orbitals
            Values: List of different active space suggestions for each active space (nel, norb).
                    Each list item is an MOListInfo typed dictionary.
        """
        # Determine a minimal space containing MOs with relevant spin contributions.
        # Unless provided, make a choice such that any adding any MOs outside the minimal space
        # cannot change the number of unpaired electrons rounded to the nearest integer.
        if minimal_space is None:
            minimal_space = self.minimal_space()
        minimal_norb = len(minimal_space)

        # Results dictionary with active space suggestions.
        spaces: ActiveSpacesDict = defaultdict(list)

        # Buffer to store the generated lists of MO indices before truncation.
        # Dictionary values are sets to avoid redundant entries.
        # Tuples are used to store MO "lists", as set items must be immutable.
        spawned_spaces: dict[int, set[tuple[int, ...]]] = defaultdict(set)

        # Start off with the minimal active space.
        spawned_spaces[minimal_norb].add(tuple(minimal_space))

        # Iterate over all numbers of orbitals to perform spawning and truncation.
        for nmo in range(minimal_norb, self.norb + 1):
            # Organize spaces to truncate for a given number of orbitals.
            # Keys:     Number of electrons.
            # Values:   List of information MOListInfo dictionaries for each MO list.
            spaces_to_truncate: dict[int, list[MOListInfo]] = defaultdict(list)
            for mo_tuple in spawned_spaces[nmo]:
                mo_info = self.info_for_mospace(mo_tuple, minimal=minimal_space)
                nel = mo_info.nel
                spaces_to_truncate[nel].append(mo_info)
            # Free up memory.
            del spawned_spaces[nmo]

            # Numbers of active electrons associated with nmo active orbitals.
            nel_with_nmo = sorted(spaces_to_truncate.keys())

            # Perform the truncation for each (nel, nmo) combination separately.
            for nel in nel_with_nmo:
                # MO list with the maximal pair sum.
                leading_space = max(spaces_to_truncate[nel], key=attrgetter("pairinfo_sum"))

                # Only keep MO lists with a pair information sum exceeding a threshold.
                # The threshold is the maximal pair information sum minus the associated decrement.
                # Always keep the MO list with the maximal pair information sum (a decrement could
                # be negative).
                thresh = leading_space.pairinfo_sum - leading_space.min_decrement
                for mo_info in spaces_to_truncate[nel]:
                    if mo_info.pairinfo_sum > thresh or mo_info is leading_space:
                        spaces[(nel, nmo)].append(mo_info)

                # Sort MO lists by their pair information sum in descending order.
                spaces[(nel, nmo)].sort(key=attrgetter("pairinfo_sum"), reverse=True)

            # Spawn new MO lists by extending lists of length nmo with one or two orbitals.
            for nel in nel_with_nmo:
                for mo_info in spaces[(nel, nmo)]:
                    mo_list = mo_info.mo_list
                    available_indices = set(range(self.norb)) - set(mo_list)

                    # Extend MO lists by one MO.
                    for i in available_indices:
                        ext_space = tuple(sorted([*mo_list, i]))
                        spawned_spaces[nmo + 1].add(ext_space)

                    # Extend MO lists by two MOs.
                    for i, k in combinations(available_indices, 2):
                        ext_space = tuple(sorted([*mo_list, i, k]))
                        spawned_spaces[nmo + 2].add(ext_space)

        spaces = self.sanitize_active_spaces(spaces)

        if mo_list_full is not None:
            spaces = map_spaces_to_full(spaces, mo_list_full)
        return spaces

    def sanitize_active_spaces(self, spaces: ActiveSpacesDict) -> ActiveSpacesDict:
        """Filter active spaces for simple sanity criteria.

        The active space must not be completely filled or empty. Moreover, number of unpaired
        electrons must be correct.

        Args:
            spaces: Dictionary with active space suggestions.

        Returns:
            Dictionary from the input with unreasonable choices removed.
        """
        sanitized_spaces: ActiveSpacesDict = {}
        for (nel, nmo), mo_lists in spaces.items():
            # Discard active spaces with zero or 2N electrons in N orbitals.
            if nel in (0, 2 * nmo):
                continue

            # the total number of electrons and the number of unpaired electrons need to
            # have the same parity.
            if (nel - self.spin) % 2 != 0:
                continue

            # Iterate over all orbital lists for (nel, nmo) spaces and perform filtering.
            filtered_spaces = []
            for mo_info in mo_lists:
                # The number of unpaired electrons needs to be correct.
                actual_spin = np.sum(self.mo_spin[mo_info.mo_list])
                if iround(actual_spin) == self.spin:
                    filtered_spaces.append(mo_info)

            # Add key / value pair if there are surviving spaces.
            if filtered_spaces:
                sanitized_spaces[(nel, nmo)] = filtered_spaces

        return sanitized_spaces

    def info_for_mospace(self, mo_list: Sequence[int], minimal: Sequence[int]) -> MOListInfo:
        """Calculate various information for a list of active orbitals.

        Args:
            mo_list:    List of active orbital indices.
            minimal:    Minimal MO space with orbitals that always need to be active. These MOs
                        are excluded when calculating minimal decrements, entropies or edge sums.

        Returns:
            Dictionary with information for the MO list provided.
        """
        # Check that the MO list contains only unique elements.
        if len(set(mo_list)) != len(mo_list):
            raise ValueError("'mo_list' contains duplicate elements.")

        if not set(mo_list) >= set(minimal):
            raise ValueError("'mo_list' must be a superset of the minimal active space.")

        # Convert the input into a sorted list of integers (may also be numpy ints etc. otherwise).
        mo_list = sorted([int(i) for i in mo_list])

        # Get the number of electrons occupying the orbitals in the MO list.
        nel = iround(np.sum(self.electrons_per_mo[mo_list]))

        # MOs are removable if they are not part of the minimal active space.
        # Exception: if the MO list is identical to the minimal space, consider all MOs removable.
        removable_mos = sorted(set(mo_list) - set(minimal))
        if not removable_mos:
            removable_mos = sorted(set(minimal))

        # Outer MOs are those orbitals that are not active in the current list.
        outer_mos = sorted(set(range(self.norb)) - set(mo_list))

        # Pair information sum over all MOs in the list.
        mo_pairsum = self.pairinfo_sum(mo_list)

        # Calculate entropy for all MOs provided to this class.
        entropy = calculate_entropy_s1(self.orbdens)

        # Calculate sum over all edges (active and inactive) for each MO provided to this class.
        edge_sum = self.pairinfo_per_orbital()

        # Calculate decrement for removable MOs in the active space. A dummy value of zero is used
        # if the list is empty.
        min_decrement = (
            min([mo_pairsum - self.pairinfo_sum(set(mo_list) - {i}) for i in removable_mos])
            if mo_list
            else 0.0
        )

        # Calculate increment for MOs outside the active space. A dummy value of zero is used if
        # all MOs are in the active space.
        max_increment = (
            max([self.pairinfo_sum([*mo_list, i]) - mo_pairsum for i in outer_mos])
            if outer_mos
            else 0.0
        )

        # Collect the results.
        return MOListInfo(
            mo_list=mo_list,
            nel=nel,
            minimal=set(mo_list) == set(minimal),
            pairinfo_sum=mo_pairsum,
            min_decrement=min_decrement,
            max_increment=max_increment,
            min_entropy=min(entropy[removable_mos]) if removable_mos else 0.0,
            min_edge_sum=min(edge_sum[removable_mos]) if removable_mos else 0.0,
        )
