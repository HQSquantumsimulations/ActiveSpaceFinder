# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
from operator import attrgetter

import pytest
from asf.asfbase import ActiveSpace
from asf.filters import ActiveSpaceSelectionError, select_max_pairinfo_sum
from asf.pairinfo import ActiveSpacesDict, MOListInfo


def find_matching_MOListInfo(unfiltered: ActiveSpacesDict, space: ActiveSpace) -> MOListInfo:
    """Find the MOListInfo object matching the given active space."""
    return next(m for group in unfiltered.values() for m in group if m.mo_list == space.mo_list)


def test_norb_nel_ethene(ethene_ASFCI):
    """Test find_one on ethene active space data with given norb and nel."""
    space_0 = ethene_ASFCI.find_one_sized(norb=6)
    assert space_0.norb == 6

    space_1 = ethene_ASFCI.find_one_sized(norb=4, nel=4)
    assert space_1.norb == 4
    assert space_1.nel == 4


def test_fallback_space(ethene_ASFCI, ethene_unfiltered_spaces, ethene_filtered_spaces):
    """Test using fallback option to find active space.

    In this case `find_many` (which yields the primary search space for `find_one`) does not
    contain a (2, 4) active space, however, the fallback collection (unfiltered spaces) does.
    """
    space = ethene_ASFCI.find_one_sized(norb=4, nel=2, fallback_unfiltered=True)
    mo_info = find_matching_MOListInfo(ethene_unfiltered_spaces, space)

    assert len(mo_info.mo_list) == 4
    assert mo_info.nel == 2
    with pytest.raises(StopIteration):
        find_matching_MOListInfo(ethene_filtered_spaces, space)


def test_default_pairinfo_sum_ordering(ethene_ASFCI, ethene_unfiltered_spaces):
    """Test returning space with largest pairinfo sum by default."""
    space_0 = ethene_ASFCI.find_one_sized(norb=4, fallback_unfiltered=True)
    mo_info_0 = find_matching_MOListInfo(ethene_unfiltered_spaces, space_0)
    assert space_0.norb == 4

    space_1 = ethene_ASFCI.find_one_sized(norb=4, nel=2, fallback_unfiltered=True)
    mo_info_1 = find_matching_MOListInfo(ethene_unfiltered_spaces, space_1)
    assert space_1.norb == 4
    assert space_1.nel == 2
    assert mo_info_0.pairinfo_sum >= mo_info_1.pairinfo_sum


def test_find_one_entropy_ethene(ethene_ASFCI, ethene_filtered_spaces):
    """Test find_one_entropy with different entropy thresholds.

    The test checks that:
    - A selection close to the default value yields a (2, 2) space for ethene.
    - A selection with a lower threshold yields a larger space.
    - Spaces are among filtered suggestions.
    """
    ENTROPY_0 = 0.07
    ENTROPY_1 = 0.14
    space_0 = ethene_ASFCI.find_one_entropy(entropy_threshold=ENTROPY_0)
    mo_info_0 = find_matching_MOListInfo(ethene_filtered_spaces, space_0)

    space_1 = ethene_ASFCI.find_one_entropy(entropy_threshold=ENTROPY_1)
    mo_info_1 = find_matching_MOListInfo(ethene_filtered_spaces, space_1)

    assert abs(mo_info_0.min_entropy - ENTROPY_0) <= abs(mo_info_1.min_entropy - ENTROPY_0)
    assert len(space_1.mo_list) < len(space_0.mo_list)
    assert mo_info_1.mo_list == [7, 8]
    assert mo_info_1.nel == 2


def test_custom_filters_ethene(ethene_ASFCI):
    """Test find_one with custom filter function.

    Two filters are applied to the ethene active spaces:
    - active orbitals must not contain MO with index 6
    - number of active electrons must be larger than 2
    """
    filters = [lambda x: 6 not in x.mo_list, lambda x: x.nel > 2]
    space = ethene_ASFCI.find_one_generic(
        filters=filters,
        selector=select_max_pairinfo_sum,
        fallback_unfiltered=True,
        fallback_minimal=False,
    )

    assert space.nel > 2
    assert 6 not in space.mo_list


def test_custom_selector_ethene(ethene_ASFCI, ethene_unfiltered_spaces):
    """Test find_one with custom selector function.

    The test defines a custom selector function that picks the active space with the largest value
    of 'min_entropy'. As a reference such a space is selected manually from the collection of
    unfiltered active spaces.
    """
    max_min_entropy = lambda _spaces: max(_spaces, key=attrgetter("min_entropy"))
    space = ethene_ASFCI.find_one_generic(
        selector=max_min_entropy, fallback_unfiltered=True, fallback_minimal=False
    )
    ref_space = max_min_entropy([m for group in ethene_unfiltered_spaces.values() for m in group])

    assert space.mo_list == ref_space.mo_list
    assert space.nel == ref_space.nel


def test_fail_on_no_result(ethene_ASFCI):
    """Test raising an exception when no space can be determined."""
    with pytest.raises(ActiveSpaceSelectionError):
        ethene_ASFCI.find_one_sized(norb=1, nel=4)


def test_find_one_entropy_fallback_modes(OH_radical_ASFCI):
    """Test fallback modes of find_one_entropy.

    Using the default entropy threshold, no active space is determined for the hydroxyl radical as
    all suggested sensible spaces have a minimal entropy below the threshold. This test asserts
    that the individual fallback mechanisms yield reasonable spaces. The expected behaviors for the
    fallback modes for this system are as follows:
      - "strict": throws an exception.
      - "below_threshold": yields the active space with the largest minimal entropy below the given
        entropy threshold. Here this is the (3, 3) space, which is thus also returned for
        subsequent fallback modes.
      - "minimal": yields the minimal active space if "strict" and "below_threshold" do not lead to
        a space being selected. Here this is achieved by setting minimal_entropy and
        entropy_threshold to the same value, so that no sensible spaces are found for "strict" and
        "below_threshold".
      - "unfiltered": yields the same result as "below_threshold".
    """
    max_norb = OH_radical_ASFCI.norb
    min_norb = 1

    # "strict"
    with pytest.raises(ActiveSpaceSelectionError):
        OH_radical_ASFCI.find_one_entropy(fallback_mode="strict")

    # below_threshold
    space_0 = OH_radical_ASFCI.find_one_entropy(fallback_mode="below_threshold")
    assert space_0.norb < max_norb
    assert space_0.norb > min_norb
    assert space_0.mo_list == [3, 4, 5]

    # minimal
    space_1 = OH_radical_ASFCI.find_one_entropy(fallback_mode="minimal", minimal_entropy=0.139)
    assert space_1.norb < max_norb
    assert space_1.mo_list == [4]

    # unfiltered
    space_2 = OH_radical_ASFCI.find_one_entropy(fallback_mode="unfiltered")
    assert space_2.norb < max_norb
    assert space_2.norb >= min_norb
    assert space_2.mo_list == space_0.mo_list

    # default ("minimal" but same result as "below_threshold")
    space_0 = OH_radical_ASFCI.find_one_entropy()
    assert space_0.norb < max_norb
    assert space_0.norb > min_norb
    assert space_0.mo_list == [3, 4, 5]
