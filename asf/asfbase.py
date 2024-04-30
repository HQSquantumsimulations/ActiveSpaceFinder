# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Core functions to calculate entropies and select orbitals."""

import sys
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from functools import partial
from io import StringIO
from math import log
from typing import Literal, Optional, Protocol, TypedDict, Union

import numpy as np
from pyscf.gto import Mole

from .filters import (
    ActiveSpaceSelectionError,
    FilterFunction,
    SelectorFunction,
    apply_filters,
    apply_selector,
    filter_nel_strict,
    filter_norb_strict,
    find_sensible,
    select_max_pairinfo_sum,
    select_min_entropy_diff,
    select_pairinfo_with_entropy,
)
from .pairinfo import ActiveSpacesDict, PairInfoAnalyzer, group_active_spaces
from .utility import ENTROPY_ZERO_CUTOFF, calculate_entropy_s1, iround

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Default parameters for the orbital selection function.
DEFAULT_ENTROPY_THRESHOLD = -0.1 * log(0.25)
DEFAULT_PLATEAU_THRESHOLD = 0.1
DEFAULT_REL_COMPARISON_THRESHOLD = 0.05
DEFAULT_CUMULANT_MINIMUM_THRESHOLD = 1.0e-6


class ASFBase(ABC):
    """Abstract base class for correlated methods to perform orbital selection.

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
    nel, mo_list = sf.entropy_selection(...)
    """

    def __init__(
        self,
        mol: Mole,
        mo_coeff: np.ndarray,
        nel: Optional[int] = None,
        norb: Optional[int] = None,
        mo_list: Optional[Union[list[int], np.ndarray]] = None,
    ) -> None:
        """Setting the initial parameters for the calculation.

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
            raise Exception("mo_coeff must by a 2-D array.")
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
                raise Exception("norb and mo_list are inconsistent.")
            self.nel = nel
            self.norb = len(mo_list)
            self.ncore = calc_ncore(self.mol, nel)
            self.mo_list = np.array(mo_list)
            if self.mo_list.ndim != 1:
                raise Exception("mo_list must be a 1-D array or a list")
        else:
            raise Exception("Incompatible combination of options nel, norb, mo_list.")

    @classmethod
    def from_preselection(cls, strategy: "Preselection", **kwargs) -> Self:
        """Determine an initial space and initialize an ASFBase class with it.

        This alternative initializer first carries out a step to determine an initial space
        following a preselection approach and then creates an ASFBase instance (e.g. ASFDMRG) with
        the MO coefficients, indices of active MOs, and number of active electrons arising from
        the initial space.

        Args:
            strategy: Approach for selecting initial active space. The ASF package provides a small
                      collection of such approaches in the `asf.preselection` module. Custom
                      preselection procedures are also supported as long as they derive from the
                      `Preselection` protocol (`asf.asfbase.Preselection`).
            **kwargs: Settings to pass on to the class initializer.

        Returns:
            A correlated calculation object derived from ASFBase instantiated with the determined
            initial active space.
        """
        space = strategy.select()
        return cls(strategy.mol, space.mo_coeff, nel=space.nel, mo_list=space.mo_list, **kwargs)

    @abstractmethod
    def calculate(self) -> None:
        """Perform a correlated calculation, which generates the information to select orbitals."""
        raise NotImplementedError

    @abstractmethod
    def one_orbital_density(self, root: int = 0) -> np.ndarray:
        """Calculate the one-orbital density.

        Args:
            root: electronic state to obtain the density for

        Returns:
            The one-orbital density as a N x 4 matrix for N orbitals.
        """
        raise NotImplementedError

    @classmethod
    def from_active_space(cls, mol: Mole, space: "ActiveSpace", **kwargs) -> Self:
        """Initialize an ASF object from an ActiveSpace instance.

        Args:
            mol:      molecule (as pyscf.gto.Mole instance)
            space:    active space
            **kwargs: Settings to pass on to the class initializer.

        Returns:
            ASFBase instance (or one derived from ASFBase)
        """
        return cls(mol, space.mo_coeff, nel=space.nel, mo_list=space.mo_list, **kwargs)

    def one_orbital_entropy(
        self, orbdens: Optional[np.ndarray] = None, root: int = 0
    ) -> np.ndarray:
        """Calculates the one-orbital entropy based on a one-orbital density.

        Args:
            orbdens: Orbital density to calculate the entropy with.
            root: electronic state to obtain the density for

        Returns:
            Vector of single-site entropies for the orbitals.
        """
        if orbdens is None:
            orbdens = self.one_orbital_density(root=root)
        return calculate_entropy_s1(orbdens)

    def entropy_selection(
        self,
        root: int = 0,
        threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        plateau_threshold: float = DEFAULT_PLATEAU_THRESHOLD,
        useCumulant: bool = True,
        verbose: bool = True,
    ) -> "ActiveSpace":
        """Deprecated: performs orbital selection via a simple entropy cutoff.

        Note that this function is considered inferior to 'find_one_entropy'. It is mainly retained
        to reproduce results of older ASF versions.

        Args:
            root:               Electronic state to perform the orbital selection for.
            threshold:          Threshold for the single-site entropies
            plateau_threshold:  Threshold to identify plateaus
            useCumulant:        Set whether to incorporate cumulant information in the selection
            verbose:            Print information or not

        Returns:
            number of electrons
            list of selected orbitals
        """
        orbdens = self.one_orbital_density(root=root)
        nel_noc, selected_mos_noc = entropy_selection(
            orbdens, threshold=threshold, plateau_threshold=plateau_threshold
        )

        if useCumulant:
            pair_cumulant = self.diagonal_cumulant(root=root)
            nel, selected_mos = entropy_selection(
                orbdens,
                pair_cumulant=pair_cumulant,
                threshold=threshold,
                plateau_threshold=plateau_threshold,
            )
        else:
            nel, selected_mos = nel_noc, selected_mos_noc

        if verbose:
            # 'a' selected via entropy, 'c' selected via entropy+cumulant
            print_mo_table(
                orbital_densities=orbdens,
                selections={
                    "a": selected_mos_noc,
                    "c": set(selected_mos) - set(selected_mos_noc),
                },
            )

        return ActiveSpace(nel=nel, mo_list=selected_mos, mo_coeff=self.mo_coeff)

    @abstractmethod
    def diagonal_cumulant(self, root: int = 0, full_space: bool = True) -> np.ndarray:
        """The 'diagonal' elements of the two-body cumulant, which derive from <p^+ q^+ q p>.

        Optionally, the matrix is filled up with zeros to cover the entire set of orbitals.

        Args:
            root: the root for which the density is computed
            full_space: if true, expand the cumulant to cover the full MO space
                        (filled up with zeros)

        Returns:
            Matrix with the relevant entries of the cumulant.
            Note: 0 labels the first active orbital.

        Raises:
            Exception: input error
        """
        raise NotImplementedError

    def unfiltered_pairinfo_spaces(self, root: int = 0) -> ActiveSpacesDict:
        """Determine a large number of candidate active spaces with very light screening.

        For each number of active electrons and active MOs, (nel, nmo), multiple active spaces are
        determined. Most of those may not be suitable for converging CASSCF calculations.

        Args:
            root:   The electronic state for which to identify active spaces.

        Returns:
            Dictionary with active space suggestions and various information as defined in
            ActiveSpacesDict. The MO lists are mapped to the full MO space.
        """
        # One-orbital density and pair information in the initial active MO subspace.
        orbdens = self.one_orbital_density(root=root)[self.mo_list, :]
        cumulant = self.diagonal_cumulant(root=root, full_space=False)

        # Determine all candidate active spaces and associated information. The sign of the
        # diagonal cumulant is inverted:
        # most energy-lowering interactions -> largest pair information.
        pa = PairInfoAnalyzer(
            pairinfo=-cumulant,
            orbdens=orbdens,
            spin=self.mol.spin,
        )
        return pa.generate_candidate_spaces(mo_list_full=self.mo_list.tolist())

    def find_many(
        self,
        root: int = 0,
        minimal_entropy: float = 0.07,
        mo_weight_type: Literal["min_entropy", "min_edge_sum", "none"] = "min_entropy",
        comparison_tolerance: float = DEFAULT_REL_COMPARISON_THRESHOLD,
        keep_minimal: bool = True,
        filters: Optional[list[FilterFunction]] = None,
    ) -> ActiveSpacesDict:
        """Selects a collection of active spaces based on pair information.

        By default `find_many` applies sensible selection criteria to a collection of unfiltered
        spaces. For a detailed description of the criteria, see `asf.filters.find_sensible`. Active
        spaces may also be filtered further following custom criteria via the `filters` argument.
        Note that the default filters can only be partially disabled by supplying
        `mo_weight_type='none'` and `minimal_entropy=0.0`.

        Args:
            root:                  Root to select the active spaces for.
            minimal_entropy:       Threshold for the minimal single-orbital entropy.
                                   Set to zero to disable.
            mo_weight_type:        Metric by which to compare orbital weights.
                                   Set to 'none' to disable.
            comparison_tolerance:  Relative tolerance to consider similar values being
                                   approximately equal.
            keep_minimal:          Whether to retain the minimal active space even if it does not
                                   satisfy filter criteria.
            filters:               A list of additional filter functions. Note that custom filter
                                   functions must follow the function signature
                                   `f(x: MOListInfo) -> bool`.
        """
        _spaces = self.unfiltered_pairinfo_spaces(root=root)
        spaces = find_sensible(
            spaces=_spaces,
            minimal_entropy=minimal_entropy,
            mo_weight_type=mo_weight_type,
            comparison_tolerance=comparison_tolerance,
            keep_minimal=keep_minimal,
            filters=filters,
        )
        return group_active_spaces(spaces)

    def find_one_generic(
        self,
        selector: Union[SelectorFunction, Sequence[SelectorFunction]],
        fallback_unfiltered: bool,
        fallback_minimal: bool,
        filters: Optional[list[FilterFunction]] = None,
        root: int = 0,
        minimal_entropy: float = 0.07,
        mo_weight_type: Literal["min_entropy", "min_edge_sum", "none"] = "min_entropy",
        comparison_tolerance: float = DEFAULT_REL_COMPARISON_THRESHOLD,
    ) -> Optional["ActiveSpace"]:
        """Find single active space based on pair information and user-defined criteria.

        This function does not make assumptions about filters/selectors (other than the criteria
        used in `find_many`) and is intended for expert users. To determine a single active space,
        `find_one_generic` first considers a pool of reasonable active spaces (using the criteria
        set out in `find_many`) that will likely converge in a CASSCF or DMRG-SCF calculation. In
        case no active space could be selected from the primary pool of candidate spaces and
        `fallback_unfiltered=True`, the same search is applied to the collection of unfiltered
        active spaces (see `unfiltered_pairinfo_spaces`). If no active space can be found with
        either method, an error will be raised.

        Args:
            root:                 Root to select the active space for.
            fallback_unfiltered:  Extends the search for an active space to the collection of
                                  unfiltered spaces. This option can be useful if the desired space
                                  is not contained in the primary set of sensible spaces. Note that
                                  spaces selected from the fallback pool may not converge in a
                                  CASSCF calculation.
            fallback_minimal:     Whether to retain the minimal space among the list of sensible
                                  spaces even if does not satisfy the filtering criteria otherwise.
            filters:              A list of filter functions. The custom filter functions must
                                  be of type `FilterFunction` defined in `asf.filters`.
            selector:             One or multiple selector functions. A selector is a custom
                                  function to pick one space from several viable choices. Note that
                                  a selector must be of type `SelectorFunction` as defined in
                                  `asf.filters`. If multiple selectors are provided, then all but
                                  the first selector act as fallback options. If the first selector
                                  fails to select one option, the second selector is applied, etc.,
                                  until one selection succeeds.
            minimal_entropy:      Threshold for the minimal single-orbital entropy.
                                  Set to zero to disable.
            mo_weight_type:       Metric by which to compare orbital weights.
                                  Set to 'none' to disable.
            comparison_tolerance: Relative tolerance to consider similar values being
                                  approximately equal.

        Returns:
            Single active space matching given criteria.
        """
        unfiltered = self.unfiltered_pairinfo_spaces(root=root)

        filter_fns: list[FilterFunction] = filters or []

        sensible = find_sensible(
            spaces=unfiltered,
            minimal_entropy=minimal_entropy,
            mo_weight_type=mo_weight_type,
            comparison_tolerance=comparison_tolerance,
            keep_minimal=fallback_minimal,
            filters=filter_fns,
        )
        selected_space = apply_selector(sensible, selector=selector)

        if (selected_space is None) and fallback_unfiltered:
            print(
                (
                    "Could not select an active space from primary collection. "
                    "Searching in unfiltered collection instead."
                )
            )
            fallback = apply_filters(unfiltered, filter_fns)
            selected_space = apply_selector(fallback, selector=selector)

        return (
            ActiveSpace(selected_space.nel, selected_space.mo_list, self.mo_coeff)
            if selected_space
            else None
        )

    def find_one_sized(
        self,
        norb: int,
        nel: Optional[int] = None,
        root: int = 0,
        fallback_unfiltered: bool = False,
        filters: Optional[list[FilterFunction]] = None,
        selector: Union[SelectorFunction, Sequence[SelectorFunction], None] = None,
        **find_many_kwargs,
    ) -> "ActiveSpace":
        """Find optimal active space of given size based on pair information and other criteria.

        To determine a single active space, `find_one_sized` first considers a pool of reasonable
        active spaces (using the criteria set out in `find_many`) that will likely converge in a
        CASSCF or DMRG-SCF calculation. In case no active space could be selected from the primary
        pool of candidate spaces and the fallback is enabled (`fallback_unfiltered=True`), the same
        search is applied to the collections of unfiltered active spaces (see
        `unfiltered_pairinfo_spaces`). If no space could be determined with either method, an error
        will be raised.

        Args:
            root:                Root to select the active space for.
            norb:                Requested number of active orbitals.
            nel:                 Requested number of active electrons.
            fallback_unfiltered: Extends the search for an active space to the collection of
                                 unfiltered spaces. This option can be useful, if the desired space
                                 is not contained in the primary set of sensible spaces. Note that
                                 spaces selected from the fallback pool may not converge in a
                                 CASSCF calculation.
            filters:             A list of additional filter functions. Note that custom filter
                                 must be of type `FilterFunction` as defined in `asf.filters`.
            selector:            A custom function or a list of custom functions to pick one space
                                 from potentially several cases. By default, the space with maximum
                                 pair information sum is returned. Note that a selector must be of
                                 type `SelectorFunction` as defined in `asf.filters`, or a list
                                 thereof. If a list is provided, the selectors are applied
                                 successively until the first one does not return None.
            find_many_kwargs:    Arguments for `find_many` excluding filters. Use these arguments
                                 to tune the number of active spaces in the primary pool of
                                 candidate spaces.

        Raises:
            ActiveSpaceSelectionError: no space found for given criteria

        Returns:
            Single active space matching given size criteria.
        """
        filter_fns: list[FilterFunction] = [partial(filter_norb_strict, norb=norb)]
        if nel is not None:
            filter_fns.append(partial(filter_nel_strict, nel=nel))
        if filters:
            filter_fns.extend(filters)

        selector_fn = selector or select_max_pairinfo_sum

        space = self.find_one_generic(
            root=root,
            filters=filter_fns,
            selector=selector_fn,
            fallback_unfiltered=fallback_unfiltered,
            fallback_minimal=True,
            **find_many_kwargs,
        )
        if not space:
            msg = "" if fallback_unfiltered else "\nConsider setting 'fallback_unfiltered=True'."
            nel_msg = f" and {nel} active electrons" if nel is not None else ""
            raise ActiveSpaceSelectionError(
                (
                    "Could not determine an active space matching "
                    f"{norb} active orbitals{nel_msg}.{msg}"
                )
            )
        return space

    def find_one_entropy(
        self,
        root: int = 0,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        fallback_mode: Literal["strict", "below_threshold", "minimal", "unfiltered"] = "minimal",
        filters: Optional[list[FilterFunction]] = None,
        **find_many_kwargs,
    ) -> "ActiveSpace":
        """Select a sensible active space based on a threshold for the minimal entropy.

        Among the `find_one` variants, this is the recommended function to select a single active
        space. To determine a single active space, `find_one_entropy` first considers a pool of
        reasonable active spaces (using the criteria set out in `find_many`) that will likely
        converge in a CASSCF or DMRG-SCF calculation. To select a single choice, only spaces with
        a minimal entropy of any orbital above the specified threshold are considered, and the
        space that maximizes the pair information sum is returned.

        Since it may occur that no active space from the primary pool matches the search criteria
        several fallback modes are provided (see argument description), where the original
        (i.e. "strict") search parameters are slightly altered to find a reasonable alternative.
        The active space searches modes of `find_one_entropy` are carried out following the order:
          "strict" -> "below_threshold" -> "minimal" -> "unfiltered".
        By default, at most the "minimal" fallback is completed before an exception is thrown.

        Args:
            root:                Root to select the active space for.
            entropy_threshold:   Entropy threshold to select a single active space; all spaces need
                                 to have a minimum value above this threshold.
            fallback_mode:       Sets the fallback behavior in case no active space could be found:
                                 - `strict` is equivalent to no fallback and raises an exception if
                                   no space matching the criteria could be found.
                                 - `below_threshold` weakens the entropy criterion and searches for
                                   sensible spaces below the given entropy threshold.
                                 - `minimal` is the default and returns the minimal active space if
                                   the previous modes did not yield a result. This can, for
                                   instance, be useful for open-shell systems.
                                 - `unfiltered` extends the search for an active space to the
                                   collection of unfiltered spaces. This option can be useful, if
                                   the desired space is not contained in the primary set of
                                   sensible spaces. Note that spaces selected from the fallback
                                   pool may not converge in a CASSCF calculation.
            filters:             A list of additional filter functions. Note that custom filter
                                 must be of type `FilterFunction` as defined in `asf.filters`.
            find_many_kwargs:    Arguments for `find_many` excluding filters. Use these arguments
                                 to tune the number of active spaces in the primary pool of
                                 candidate spaces.

        Returns:
            Single active space with minimal entropy above the specified threshold and
            with maximal pair information sum.
        """
        fallback_below_thresh = fallback_mode != "strict"
        fallback_minimal = fallback_mode in ("minimal", "unfiltered")
        fallback_unfiltered = fallback_mode == "unfiltered"

        # 1) The first selector filters out all spaces that have at least one MO below the
        #    threshold. Among the surviving spaces it chooses the one that maximizes the pair
        #    information sum. The minimal space is filtered out if it contains orbitals with an
        #    entropy below the threshold.
        # 2) The second selector (activated if the fallback mode is 'below_threshold') will be used
        #    if step one did not yield any results; therefore, it will select the highest entropy
        #    below the threshold. If multiple spaces have got a numerically identical minimal
        #    entropy, then the space that maximizes the pair information sum will be chosen. This
        #    can happen if MOs with lower entropies are included through the pair information
        #    criterion in smaller spaces, and orbitals with larger entropies are left over to be
        #    included in larger spaces.
        # 3) If the fallback mode was specified as 'minimal', then the filtered set of sensible
        #    spaces inside find_one_generic will also contain the minimal space. In case that no
        #    space exists with a higher minimal entropy than the minimal space, the second selector
        #    will return the minimal space.
        # 4) If the fallback mode is 'unfiltered', the two selectors will be applied to the spaces
        #    without filtering in case the selection attempt with filtered spaces was unsuccessful.
        #    That means an attempt will be made to select an unfiltered space with a minimal
        #    entropy above the threshold that maximizes the pair information sum; if that fails,
        #    the space with the highest minimum entropy below the threshold will be returned.
        selector_list: list[SelectorFunction] = [
            partial(
                select_pairinfo_with_entropy,
                entropy_threshold=entropy_threshold,
                keep_minimal=False,
            ),
        ]
        if fallback_below_thresh:
            selector_list.append(
                partial(select_min_entropy_diff, target_entropy=entropy_threshold, rel_tol=0.0)
            )

        space = self.find_one_generic(
            root=root,
            fallback_unfiltered=fallback_unfiltered,
            fallback_minimal=fallback_minimal,
            filters=filters,
            selector=selector_list,
            **find_many_kwargs,
        )

        if not space:
            raise ActiveSpaceSelectionError(
                (
                    f"Could not determine an active space with "
                    f"{entropy_threshold=:.3f} and {fallback_mode=}.\n"
                    "Consider adjusting the search parameters and/or the fallback mode."
                )
            )

        return space

    @abstractmethod
    def rdm1s(self, root: int = 0) -> np.ndarray:
        """Returns the active one-electron spin density matrix as an 2 x N x N array."""
        raise NotImplementedError

    @abstractmethod
    def rdm2(self, root: int = 0) -> np.ndarray:
        """Returns the active two-electron spin-free density matrix as a four-dimensional array."""
        raise NotImplementedError


class ActiveSpace:
    """Class to store information about an active space.

    Attributes:
        nel:        Number of active electrons.
        norb:       Number of active orbitals.
        mo_list:    List of active MO indices (referring to columns of mo_coeff).
        mo_coeff:   Matrix of all MO coefficients (active and inactive).
    """

    nel: int
    norb: int
    mo_list: list[int]
    mo_coeff: np.ndarray

    class DataDict(TypedDict):
        """Serialized data of the attributes in an ActiveSpace object."""

        nel: int
        mo_list: list[int]
        mo_coeff: np.ndarray

    def __init__(
        self, nel: int, mo_list: Union[list[int], np.ndarray], mo_coeff: np.ndarray
    ) -> None:
        """Set up the object with information about an active space.

        Args:
            nel:        Number of active electrons.
            mo_list:    List of active MO indices (referring to columns of mo_coeff).
            mo_coeff:   Matrix of all MO coefficients (active and inactive).
        """
        self.nel = nel
        self.mo_coeff = mo_coeff
        self.mo_list = list(map(int, sorted(set(mo_list))))
        self.norb = len(mo_list)

        self._validate_nel()
        self._validate_mo_list()
        self._validate_mo_coeff()

    def _validate_nel(self) -> None:
        """Perform sanity checks on the number of active electrons.

        Raises:
            ValueError: Invalid number of electrons
        """
        if self.nel < 0:
            raise ValueError("Invalid number of electrons! Expected a positive integer.")

    def _validate_mo_coeff(self) -> None:
        """Perform sanity checks on the MO coefficient matrix.

        Raises:
            ValueError: bad dimensions
        """
        if self.mo_coeff.ndim != 2:
            raise ValueError("mo_coeff must be a matrix (array with dimension two).")
        if self.mo_coeff.shape[1] > self.mo_coeff.shape[0]:
            raise ValueError("More MOs than AOs. Bad MO coefficient matrix?")

    def _validate_mo_list(self) -> None:
        """Perform sanity checks on the active MO list.

        Raises:
            ValueError: inconsistency within or with mo_coeff
        """
        if min(self.mo_list, default=0) < 0:
            raise ValueError("Negative index in MO list!")
        if max(self.mo_list, default=0) >= self.mo_coeff.shape[1]:
            raise ValueError("MO list indices outside the bounds of the MO coefficient matrix.")

    def from_active_indices(self, mo_list: list[int]) -> list[int]:
        """Map relative MO indices within the active space onto the full set of MOs.

        Args:
            mo_list:  List of relative indices within the active space (relative to self.mo_list).

        Returns:
            mo_list mapped to the full set of orbitals (= relative to the columns of mo_coeff).
        """
        return [self.mo_list[i] for i in mo_list]

    def to_active_indices(self, mo_list: list[int]) -> list[int]:
        """Map MO indices onto relative indices within the active space.

        Args:
            mo_list:  List of indices referring to the full set of MOs (= columns of mo_coeff).

        Returns:
            mo_list mapped relative to self.mo_list (= relative to the active space indices).
        """
        return [self.mo_list.index(i) for i in mo_list]

    def to_dict(self) -> "ActiveSpace.DataDict":
        """Serialize the data in the attributes of this class.

        Returns:
            A dictionary containing the needed data to reproduce this object.
        """
        data_dict: "ActiveSpace.DataDict" = {
            "nel": self.nel,
            "mo_list": self.mo_list,
            "mo_coeff": self.mo_coeff,
        }
        return data_dict

    @classmethod
    def from_dict(cls, data: "ActiveSpace.DataDict") -> "ActiveSpace":
        """Create a new ActiveSpace object from a dictionary of serialized data.

        Args:
            data:   Data as returned by self.to_dict().

        Raises:
            Exception: cannot instantiate with given data

        Returns:
            A new instance of this class.
        """
        try:
            return cls(**data)
        except Exception as err:
            raise err

    def merge_with(
        self,
        mol: Mole,
        other: Optional["ActiveSpace"] = None,
        mo_coeff_rtol: float = 1e-5,
        mo_coeff_atol: float = 1e-8,
    ) -> "ActiveSpace":
        """Merge two active spaces based on the same MOs.

        This function is based on the following assumptions:
            - A sensible active space can be obtained simply by merging mo_list1 and mo_list2.
            - Orbitals that are not already active in both active space definitions "bring" either
              two electrons (occup. MO) or no electrons (virtual MO) into the merged active space.
            - Inactive MOs are always ordered: doubly occupied MOs come before virtual orbitals.
        For further details, see `merge_active_spaces`.

        Raises:
            ValueError: MO coefficient matrices do not match

        Args:
            mol:    Mole object with molecular data.
            other:  Other ActiveSpace instance. If not provided, the current instance is returned.
            mo_coeff_rtol: Relative tolerance for comparing a pair of MO coefficient matrices.
            mo_coeff_atol: Absolute tolerance for comparing a pair of MO coefficient matrices.
        """
        if other is None:
            return self
        if not np.allclose(self.mo_coeff, other.mo_coeff, rtol=mo_coeff_rtol, atol=mo_coeff_atol):
            raise ValueError("Cannot merge active spaces. MO coefficients do not match!")
        nel, mo_list = merge_active_spaces(
            mol=mol, nel1=self.nel, mo_list1=self.mo_list, nel2=other.nel, mo_list2=other.mo_list
        )
        return ActiveSpace(nel=nel, mo_list=mo_list, mo_coeff=self.mo_coeff)

    def __repr__(self) -> str:
        """Provide basic active space data in a compact format."""
        return f"ActiveSpace(nel={self.nel}, mo_list={self.mo_list}, mo_coeff=...)"

    def __str__(self) -> str:
        """Provide basic active space data in a user-friendly format."""
        return (
            f"ActiveSpace: {self.nel} electrons in {self.norb} orbitals\n"
            f"  active MO indices: {self.mo_list}"
        )


class Preselection(Protocol):
    """A general class interface for preselection strategies."""

    def select(self) -> ActiveSpace:
        """Perform the preselection.

        Returns:
            An active space as an ActiveSpace instance
        """
        ...

    @property
    def mol(self) -> Mole:
        """Molecule associated with the preselection and active space search."""
        ...


def print_mo_table(
    orbital_densities: np.ndarray,
    mo_list: Optional[list[int]] = None,
    selections: Optional[Union[list[str], dict[str, Collection[int]]]] = None,
) -> None:
    """Print an MO table with one-orbital densities, entropies and whether they are selected.

    Note, that if selections are passed as a dictionary, the collections are not checked for
    overlapping indices.

    Args:
        orbital_densities: single-orbital densities as N x 4 matrix.
        mo_list: List of indices to print. If no list is provided, all MOs will be printed
        selections: Indices of selected MO and their corresponding status labels
                    OR a list of status labels.
    """
    n_orb = len(mo_list) if mo_list else orbital_densities.shape[0]
    status: list[str] = []
    if isinstance(selections, list):
        status = selections
    elif isinstance(selections, dict):
        status = [" "] * orbital_densities.shape[0]
        for code, selected in selections.items():
            for mo in selected:
                status[mo] = code
        if mo_list:
            status = [status[i] for i in mo_list]

    if status and len(status) != n_orb:
        print("Cannot print MO table! Status list length does not match number of requested MOs.")
        return
    entropy = calculate_entropy_s1(orbital_densities)

    stream = StringIO()
    stream.write(" MO    w( )    w(\u2191)    w(\u2193)    w(\u21c5)       S")
    if status:
        stream.write("  sel")
    stream.write("\n")

    for i in range(n_orb):
        idx = mo_list[i] if mo_list else i
        stream.write(
            "{0:3d}  {1:6.3f}  {2:6.3f}  {3:6.3f}  {4:6.3f}  {5:6.3f}".format(
                idx,
                orbital_densities[idx, 0],
                orbital_densities[idx, 1],
                orbital_densities[idx, 2],
                orbital_densities[idx, 3],
                entropy[idx],
            )
        )
        if status:
            stream.write(f"  {status[i]:>3s}")
        stream.write("\n")
    print(stream.getvalue())


def calc_ncore(mol: Mole, nel_act: int) -> int:
    """Calculates the number of inactive / core orbitals from the number of active electrons.

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
        raise ValueError("Obtained an odd number of core electrons.")
    return ncoreelec // 2


def inactive_orbital_lists(
    ncore: int, nmo: int, active_list: Union[list[int], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Obtain lists of inactive core and virtual orbitals from an active space specification.

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
        raise ValueError("Active space definition is inconsistent with the number of MOs.")

    # partition inactive orbitals into core and virtuals
    core_orbitals = inactive_orbitals[:ncore]
    virtual_orbitals = inactive_orbitals[ncore:]
    return core_orbitals, virtual_orbitals


def entropy_selection(
    orbdens: np.ndarray,
    pair_cumulant: Optional[np.ndarray] = None,
    threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    plateau_threshold: float = DEFAULT_PLATEAU_THRESHOLD,
    zero_cutoff: float = ENTROPY_ZERO_CUTOFF,
    cumulant_minimum_threshold: float = DEFAULT_CUMULANT_MINIMUM_THRESHOLD,
) -> tuple[int, list[int]]:
    """Select orbitals with entropies above a given threshold.

    In addition, the list of chosen orbitals is extended
    - until the number of unpaired electrons is correct,
    - until the separation between entropies is sufficiently large ('plateaus'),
    - and with orbitals which are predominantly singly occupied.

    Args:
        orbdens:                one-orbital density with
                                columns representing empty, spin-up, spin-down, doubly occupied
        pair_cumulant:          If provided, incorporate cumulant information in the orbital
                                selection.
        threshold:              Normal threshold for the entropy (unless there is a plateau)
        plateau_threshold:      Relative threshold to identify 'plateaus'
        zero_cutoff:            Threshold to neglect zeroes when calculating the entropy
        cumulant_minimum_threshold: Do not consider cumulant information below a 'noise' cutoff

    Returns:
        number of electrons
        list of orbitals

    Raises:
        Exception: various errors
    """
    if not ((orbdens.ndim == 2) and (orbdens.shape[1] == 4)):
        raise Exception("Wrong orbital density dimensions.")

    norb = orbdens.shape[0]
    entropies = calculate_entropy_s1(orbdens, zero_cutoff)

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
        for i in entropy_sort[len(selected) :]:
            if entropies[i] >= entropies[selected[-1]] * (1.0 - plateau_threshold):
                selected.append(i)
            else:
                break

    # identify orbitals that are predominantly singly occupied (spin up or spin down)
    for i in range(norb):
        if orbdens[i, 1] + orbdens[i, 2] > orbdens[i, 0] + orbdens[i, 3]:
            if i not in selected:
                selected.append(i)

    # for each orbital the has already been selected, also include the orbital that
    # it has the largest diagonal/pair cumulant with
    if pair_cumulant is not None:
        for i in selected.copy():
            k = np.argmax(abs(pair_cumulant[i, :]))
            if (k not in selected) and (abs(pair_cumulant[i, k]) > cumulant_minimum_threshold):
                selected.append(k)

    selected.sort()

    # count the total number of electrons
    orbital_electrons = orbdens[:, 1] + orbdens[:, 2] + 2.0 * orbdens[:, 3]
    electrons_total = sum(orbital_electrons[selected])

    return iround(electrons_total), selected


def all_orbital_selections(orbdens: np.ndarray, **kwargs) -> dict[tuple[int, int], list[int]]:
    """Generate a list of all active spaces that can be generated by varying the entropy threshold.

    Args:
        orbdens:    one-orbital density in an N x 4 array
        **kwargs:   keyword arguments for the entropy_selection function

    Returns:
        Dictionary containing all the different active space suggestions.
        The keys are tuples containing (no. of electrons, no. of orbitals).
        The values are the lists of active orbitals.

    Raises:
        ValueError: invalid arguments supplied
    """
    if not ((orbdens.ndim == 2) and (orbdens.shape[1] == 4)):
        raise ValueError("Wrong orbital density dimensions.")

    norb = orbdens.shape[0]

    if "threshold" in kwargs:
        raise ValueError("Entropy threshold must not be provided to determine grouped orbitals.")

    # determine entropies sorted in ascending order
    zero_cutoff = kwargs.get("zero_cutoff", None) or ENTROPY_ZERO_CUTOFF
    entropies = calculate_entropy_s1(orbdens, zero_cutoff)
    entropies_sorted = np.sort(entropies)

    # Naive implementation: wrap around the entropy_selection function, using thresholds between
    # the entropies of orbitals.
    active_spaces = {}
    for i in range(1, norb):
        threshold = (entropies_sorted[i - 1] + entropies_sorted[i]) / 2
        nel, mo_list = entropy_selection(orbdens, threshold=threshold, **kwargs)
        active_spaces[(nel, len(mo_list))] = mo_list

    return active_spaces


def merge_active_spaces(
    mol: Mole, nel1: int, mo_list1: list[int], nel2: int, mo_list2: list[int]
) -> tuple[int, list[int]]:
    """Perform a simple merging of active spaces.

    This function attempts to construct an active space for state-averaged CASSCF calculations, by
    forming the union of the lists of active orbitals for individual electronic states.

    This function is based on the following assumptions:
        - A sensible active space can be obtained simply by merging mo_list1 and mo_list2.
        - Orbitals that are not already active in both active space definitions "bring" either two
          electrons (occupied MO) or no electrons (virtual MO) into the merged active space.
        - Inactive MOs are always ordered: doubly occupied orbitals come before virtual orbitals.

    Args:
        mol:        Mole object with molecular data.
        nel1:       Number of electrons in the first active space.
        mo_list1:   List of orbitals in the first active space.
        nel2:       number of electrons in the second active space.
        mo_list2:   List of orbitals in the second active space.

    Returns:
        Tuple representing the merged active space: (number of electrons, list of orbitals)

    Raises:
        ValueError: Insonsistent input provided.
    """
    mo_set1 = set(mo_list1)
    mo_set2 = set(mo_list2)

    if len(mo_set1) != len(mo_list1) or len(mo_set2) != len(mo_list2):
        raise ValueError("Found duplicate orbital indices in active orbitals list.")

    if nel1 > 2 * len(mo_set1) or nel2 > 2 * len(mo_set2):
        raise ValueError("Too many electrons to be accomodated in the active orbitals.")

    if nel1 == 0 and not mo_list1:
        return nel2, mo_list2
    elif nel2 == 0 and not mo_list2:
        return nel1, mo_list1

    # The merged set of active spaces.
    mo_merged = set.union(mo_set1, mo_set2)

    # Number of occupied ("core") orbitals associated with each active space.
    ncore1 = calc_ncore(mol, nel1)
    ncore2 = calc_ncore(mol, nel2)

    # Set of core orbitals associated with the first active space.
    core_set1: set[int] = set()
    ctr = 0
    while len(core_set1) < ncore1:
        if ctr not in mo_set1:
            core_set1.add(ctr)
        ctr += 1

    # Set of core orbitals associated with the second active space.
    core_set2: set[int] = set()
    ctr = 0
    while len(core_set2) < ncore2:
        if ctr not in mo_set2:
            core_set2.add(ctr)
        ctr += 1

    # Determine the number of electrons in the merged active space.
    nel = nel1 + 2 * len(core_set1.intersection(mo_merged))

    # Check consistency of the number of electrons.
    if nel2 + 2 * len(core_set2.intersection(mo_merged)) != nel:
        raise ValueError("Active space specifications are inconsistent.")

    return nel, sorted(mo_merged)
