# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Functions to filter and select from active spaces."""

from collections.abc import Sequence
from functools import partial
from math import isclose, log
from operator import attrgetter
from typing import Any, Literal, Optional, Protocol, Union

from .pairinfo import ActiveSpacesDict, MOListInfo, less_or_equal


class ActiveSpaceSelectionError(Exception):
    """ASF Exception related to selecting active spaces."""


class FilterFunction(Protocol):
    """General function interface for active space filter functions.

    Given a set of active spaces (here represented as `MOListInfo` objects) it is often desired to
    create a subset following certain constraints. These constraints can be expressed in the form
    of filter functions as defined below. A subset can then be easily created via Python's built-in
    `filter()`.
    Note: due to a more complex function signature and limitations of `Callable` in Python 3.9,
    the function type is defined as a callback protocol instead.
    """

    def __call__(self, space: MOListInfo, *args: Any, **kwargs: Any) -> bool:
        """Runs filter function with at least one mandatory MOListInfo argument."""
        ...


class SelectorFunction(Protocol):
    """General function interface for active space selector functions.

    Conceptually, selector functions ensure that only a single active space (or none) is picked
    from a set of active spaces (here represented as M̀OListInfo` objects). Hence, selectors often
    entail an optimization problem, i.e. finding the active space that optimizes an objective
    function. Moreover, they can involve filtering functionality prior to the selection.
    Note: due to a more complex function signature and limitations of `Callable` in Python 3.9,
    the function type is defined as a callback protocol instead.

    We may encounter a situation where a selection fails, either due to an empty list of active
    spaces, or due to an unsatisfied selection criterion. In that case, the selector function is
    permitted to return None.
    """

    def __call__(
        self, spaces: list[MOListInfo], *args: Any, **kwargs: Any
    ) -> Optional[MOListInfo]:
        """Runs selector function with at least one mandatory MOListInfo array argument."""
        ...


def apply_filters(
    spaces: Union[list[MOListInfo], ActiveSpacesDict],
    filters: list[FilterFunction],
) -> list[MOListInfo]:
    """Apply several filter functions to each element of an active spaces collection.

    Note that if an `ActiveSpacesDict` is passed, the filtering will not occur per CAS tuple group
    but on the entire collection of active spaces as a flat list.

    Args:
        spaces:   List of active spaces (as MOListInfo).
        filters:  List of filter functions that follow the function signature
                  `f(x: MOListInfo) -> bool`.

    Raises:
        TypeError: invalid active space collection type

    Returns:
        A list of active spaces matching the given criteria
    """
    if isinstance(spaces, dict):
        _spaces = [mo_info for mo_group in spaces.values() for mo_info in mo_group]
    elif isinstance(spaces, list):
        _spaces = spaces
    else:
        raise TypeError("Invalid type of active space collection.")

    def apply_all_filters(space: MOListInfo) -> bool:
        """Apply all filter functions for a given element."""
        return all(f(space) for f in filters)

    return list(filter(apply_all_filters, _spaces))


def apply_selector(
    spaces: Union[list[MOListInfo], ActiveSpacesDict],
    selector: Union[SelectorFunction, Sequence[SelectorFunction]],
) -> Optional[MOListInfo]:
    """Apply a selector to select at most a single element from an active spaces collection.

    If multiple selectors are provided, then the first selector is applied initially. In case it
    fails to select an active space and returns None, the second selector is applied. This is
    continued until either a single active space has been selected, or the sequence of selectors
    has been exhausted.

    Note that if an `ActiveSpacesDict` is passed, it will be converted to a flat list prior to the
    selection step.

    Args:
        spaces:     Active spaces as a list of MOListInfo objects, or as an ActiveSpacesDict.
        selector:   Individual selector function or a sequence of selector functions with the
                    signature `f(x: list[MOListInfo]) -> Optional[MOListInfo]`. If multiple
                    selectors are provided, their priority is determined by their ordering.

    Returns:
        Object with the selected active space, or None if nothing could be selected.
    """
    # Nothing to select from if an empty object was provided.
    if not spaces:
        return None

    # Ensure that the spaces are stored in a flat list.
    if isinstance(spaces, dict):
        spaces_list = [mo_info for mo_group in spaces.values() for mo_info in mo_group]
    elif isinstance(spaces, list):
        spaces_list = spaces
    else:
        raise TypeError("Invalid type of active space collection.")

    # Pack a single selector into a list.
    selector_seq = selector if isinstance(selector, Sequence) else [selector]

    for next_selector in selector_seq:
        if (selected := next_selector(spaces_list)) is not None:
            return selected
    return None


def filter_entropy_above_thresh(
    space: MOListInfo, threshold: float, keep_minimal: bool = False
) -> bool:
    """Filter active spaces that have a minimal entropy above a given threshold.

    Note that since the entropy is non-negative, also given the threshold must be larger than
    zero. If that is not the case, the filter is essentially disabled by returning `True` for each
    space. This function is supposed to be used in a list(filter(...)) statement.

    Args:
        space:        An element from a list of active spaces (as MOListInfo).
        threshold:    Threshold for the smallest single-orbital entropies.
        keep_minimal: Whether to retain the minimal active space in the filtered set.
    """
    if threshold <= 0.0:
        return True
    return space.min_entropy >= threshold or (keep_minimal and space.minimal)


def filter_node_weight_isclose(
    space: MOListInfo,
    weight_type: Literal["min_entropy", "min_edge_sum"],
    target_value: float,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-6,
) -> bool:
    """Filter active spaces that have a 'node weight' close to a given target value.

    The 'node weights', i.e. orbital-assignable metrics, can be either the minimal single-orbital
    entropy (`min_entropy`) or the minimum sum over all edges (`min_edge_sum`).

    Args:
        space:          An element from a list of active spaces (as MOListInfo).
        weight_type:    Use entropies or edge sums for comparison.
        target_value:   Target value to compare with node weight.
        rel_tol:        Relative comparison tolerance between node weight values.
        abs_tol:        Absolute comparison tolerance between node weight values.
    """
    return isclose(getattr(space, weight_type), target_value, rel_tol=rel_tol, abs_tol=abs_tol)


def filter_increments(
    space: MOListInfo,
    keep_minimal: bool,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-6,
) -> bool:
    """Filter active spaces by increment and decrement information.

    Active spaces are only retained if the largest increment is smaller than or approximately equal
    to the smallest decrement. This function is supposed to be used in a list(filter(...))
    statement.

    Args:
        space:          An element from a list of active spaces (as MOListInfo).
        keep_minimal:   Whether to keep a space flagged as minimal.
        rel_tol:        Relative comparison tolerance between increments and decrements.
        abs_tol:        Absolute comparison tolerance between increments and decrements.
    """
    return less_or_equal(
        space.max_increment, space.min_decrement, rel_tol=rel_tol, abs_tol=abs_tol
    ) or (keep_minimal and space.minimal)


def filter_norb_strict(space: MOListInfo, norb: int) -> bool:
    """Filter active spaces that strictly have a given number of MOs.

    This function is supposed to be used in a list(filter(...)) statement.

    Args:
        space: An element from a list of active spaces (as MOListInfo).
        norb:  Number of active orbitals the desired space should have.
    """
    return len(space.mo_list) == norb


def filter_nel_strict(space: MOListInfo, nel: int) -> bool:
    """Filter active spaces that strictly have a given number of MOs.

    This function is supposed to be used in a list(filter(...)) statement.

    Args:
        space: An element from a list of active spaces (as MOListInfo).
        nel:   Number of active orbitals the desired space should have.
    """
    return space.nel == nel


def select_max_pairinfo_sum(spaces: list[MOListInfo]) -> Optional[MOListInfo]:
    """Select the active space with maximum pair information sum.

    Args:
        spaces: List of active spaces (as MOListInfo).

    Returns:
        The selected space, or None.
    """
    return max(spaces, key=attrgetter("pairinfo_sum"), default=None)


def select_pairinfo_with_entropy(
    spaces: list[MOListInfo], entropy_threshold: float, keep_minimal: bool
) -> Optional[MOListInfo]:
    """This selector combines entropy threshold filtering with pairinfo maximization.

    First, it filters the spaces such that only those with a minimal entropy above a threshold are
    left. Among the surviving spaces, it selects the one that contains the largest pair information
    sum.

    Args:
        spaces:             List of active spaces (as MOListInfo) to select from.
        entropy_threshold:  Minimum entropy threshold used to filter out spaces.
        keep_minimal:       If set to True, the minimal space will always be retained in filtering
                            regardless of its orbital entropies. If set to False, it will be
                            filtered out if entropies are too low.
    """
    filtered_spaces = apply_filters(
        spaces,
        [
            partial(
                filter_entropy_above_thresh, threshold=entropy_threshold, keep_minimal=keep_minimal
            )
        ],
    )
    return select_max_pairinfo_sum(filtered_spaces)


def select_min_entropy_diff(
    spaces: list[MOListInfo],
    target_entropy: float,
    rel_tol: float,
) -> Optional[MOListInfo]:
    """Select active space that has minimum entropy closest to a target entropy.

    The "distance" between target entropy and minimum entropy of a space is calculated as the
    absolute logarithm of their ratio. Because it is quite likely that the spaces contain two or
    more active spaces with identical values of `min_entropy`, the pair information sum is used as
    a secondary criterion to pick one active space that has a value of `min_entropy` closest to
    the target entropy and the maximum pair information sum. Active spaces with nearly identical
    values of `min_entropy` are identified using `math.isclose`.

    Args:
        spaces:         List of active spaces (as MOListInfo).
        target_entropy: Target single-orbital entropy to find active space with closest
                        minimum single-orbital entropy ('min_entropy').
        rel_tol:        Relative tolerance to determine whether an entropy is close to the target
                        entropy.

    Returns:
        Representation of the selected active space.

    Raises:
        ValueError: invalid entropy value
    """
    if target_entropy <= 0.0:
        raise ValueError("Target entropy must be a positive number.")

    if not spaces:
        return None

    sorted_by_entropy = sorted(spaces, key=lambda x: abs(log(x.min_entropy / target_entropy)))
    closest = sorted_by_entropy[0].min_entropy

    return max(
        [m for m in sorted_by_entropy if isclose(m.min_entropy, closest, rel_tol=rel_tol)],
        key=attrgetter("pairinfo_sum"),
    )


def screen_node_weights(
    spaces: ActiveSpacesDict,
    weight_type: Literal["min_entropy", "min_edge_sum"],
    rel_tol: float,
    abs_tol: float = 1e-6,
) -> ActiveSpacesDict:
    """Screen active spaces for their minimal 'node weight' in each active space.

    The 'node weights' can be either entropies or sums over all edges (to active and inactive MOs).
    Active spaces are not inspected individually, but in comparison with all other active spaces
    with the same number of active electrons and orbitals. The active space with the largest
    minimal node weight is kept, and in addition also active spaces if their minimum node weight is
    sufficiently close.

    Args:
        spaces:         Dictionary containing active spaces with associated information.
        weight_type:    Use entropies or edge sums for comparison.
        rel_tol:        Relative comparison tolerance between node weight values.
        abs_tol:        Absolute comparison tolerance between node weight values.

    Returns:
        Dictionary of active spaces that has been filtered according to the criteria.
    """
    filtered_dict = {}
    for key, group in spaces.items():
        threshold = getattr(max(group, key=attrgetter(weight_type)), weight_type)
        filter_fn = partial(
            filter_node_weight_isclose,
            weight_type=weight_type,
            target_value=threshold,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        filtered_list = list(filter(filter_fn, group))
        if filtered_list:
            filtered_dict[key] = filtered_list
    return filtered_dict


def find_sensible(
    spaces: ActiveSpacesDict,
    minimal_entropy: float,
    mo_weight_type: Literal["min_entropy", "min_edge_sum", "none"],
    comparison_tolerance: float,
    keep_minimal: bool = False,
    filters: Optional[list[FilterFunction]] = None,
) -> list[MOListInfo]:
    """Selects a collection of reasonable active spaces based on pair information.

    By default the following criteria are used:
        1) For each group of active spaces with same number of active electrons and active
           orbitals, (nel, nmo), the active space must include the orbitals with the largest
           node weights (to within a tolerance).
        2) The smallest entropy of any orbital in the active space must be above the threshold.
        3) The smallest decrement must exceed the largest increment.

    The spaces can additionally be filtered according to custom criteria by passing filter
    functions via the `filters` argument. Note that when using custom filters, criteria 1), 2),
    and 3) are still applied. If a custom filtering without criteria 1-3) is desired, use
    `apply_filters` instead.

    Args:
        spaces:                Dictionary containing active spaces with associated information.
        minimal_entropy:       Threshold for criterion (2). Set to zero to disable.
        mo_weight_type:        Metric by which to compare orbital weights.
        comparison_tolerance:  Relative tolerance to establish equality.
        keep_minimal:          Whether to retain the minimal active space even if the filter
                               criteria do not yield any space.
        filters:               A list of custom filter functions. Note that filter functions must
                               follow the function signature `f(x: MOListInfo) -> bool`.

    Returns:
        List of active space suggestions.
    """
    if mo_weight_type != "none":
        spaces = screen_node_weights(spaces, mo_weight_type, comparison_tolerance)

    filter_fns: list[FilterFunction] = [
        partial(filter_entropy_above_thresh, threshold=minimal_entropy, keep_minimal=keep_minimal),
        partial(filter_increments, rel_tol=comparison_tolerance, keep_minimal=keep_minimal),
    ]
    if filters:
        filter_fns.extend(filters)

    return apply_filters(spaces, filter_fns)
