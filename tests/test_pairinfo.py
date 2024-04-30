# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import random
from math import isclose

import numpy as np
import pytest

from asf.pairinfo import (
    ActiveSpacesDict,
    MOListInfo,
    PairInfoAnalyzer,
    cumulant2b,
    cumulant2bs,
    less_or_equal,
    somo_from_orbdens,
)


def test_cumulant2b_cumulant2bs():
    for n in 1, 2, 3, 5, 10:
        # construct fake, random RDMs with correct permutation properties
        rdm2aa = np.random.random((n, n, n, n))
        rdm2aa = rdm2aa - rdm2aa.transpose((2, 1, 0, 3))
        rdm2aa = rdm2aa - rdm2aa.transpose((0, 3, 2, 1))
        rdm2aa = rdm2aa + rdm2aa.transpose((1, 0, 3, 2))
        rdm2ab = np.random.random((n, n, n, n))
        rdm2ab = rdm2ab + rdm2ab.transpose((1, 0, 3, 2))
        rdm2bb = np.random.random((n, n, n, n))
        rdm2bb = rdm2bb - rdm2bb.transpose((2, 1, 0, 3))
        rdm2bb = rdm2bb - rdm2bb.transpose((0, 3, 2, 1))
        rdm2bb = rdm2bb + rdm2bb.transpose((1, 0, 3, 2))

        rdm1a = np.einsum("pqrr->pq", rdm2aa) + np.einsum("pqrr->pq", rdm2ab)
        rdm1b = np.einsum("pqrr->pq", rdm2bb) + np.einsum("rrpq->pq", rdm2ab)

        rdm2 = rdm2aa + rdm2ab + rdm2ab.transpose((2, 3, 0, 1)) + rdm2bb

        # calculate spin components of the cumulant
        c2aa, c2ab, c2bb = cumulant2bs(rdm1a, rdm1b, rdm2aa, rdm2ab, rdm2bb)
        c2 = cumulant2b(rdm1a, rdm1b, rdm2)

        # compare sum of spin components with spin-free cumulant
        c2sum = c2aa + c2ab + c2ab.transpose((2, 3, 0, 1)) + c2bb
        assert np.allclose(c2sum, c2, atol=1e-12, rtol=0)

        # check individual numbers with random indices
        for _ in range(10):
            p, q, r, s = np.random.randint(n, size=4)
            c2aa_ref = rdm2aa[p, q, r, s] - rdm1a[p, q] * rdm1a[r, s] + rdm1a[p, s] * rdm1a[r, q]
            assert isclose(c2aa[p, q, r, s], c2aa_ref, abs_tol=1e-12, rel_tol=0.0)

            p, q, r, s = np.random.randint(n, size=4)
            c2ab_ref = rdm2ab[p, q, r, s] - rdm1a[p, q] * rdm1b[r, s]
            assert isclose(c2ab[p, q, r, s], c2ab_ref, abs_tol=1e-12, rel_tol=0.0)

            p, q, r, s = np.random.randint(n, size=4)
            c2bb_ref = rdm2bb[p, q, r, s] - rdm1b[p, q] * rdm1b[r, s] + rdm1b[p, s] * rdm1b[r, q]
            assert isclose(c2bb[p, q, r, s], c2bb_ref, abs_tol=1e-12, rel_tol=0.0)


def test_somo_from_orbdens():
    for ndomo in range(5):
        for nsomo in range(5):
            for nvirt in range(5):
                if ndomo + nsomo + nvirt == 0:
                    continue

                orbdens = np.array(
                    [[0.0, 0.0, 0.0, 1.0]] * ndomo
                    + [[0.0, 1.0, 0.0, 0.0]] * nsomo
                    + [[1.0, 0.0, 0.0, 0.0]] * nvirt
                )
                somo_list = list(range(ndomo, ndomo + nsomo))
                assert somo_from_orbdens(orbdens) == somo_list

                np.random.shuffle(orbdens)
                somo_list = list(np.nonzero(orbdens[:, 1])[0])
                assert somo_from_orbdens(orbdens) == somo_list


def test_somo_from_orbdens_fail():
    # Wrong shape.
    with pytest.raises(ValueError):
        somo_from_orbdens(np.random.random((5, 5)))

    # More beta electrons than alpha electrons.
    with pytest.raises(ValueError):
        somo_from_orbdens(
            np.array(
                [
                    [0.1, 0.0, 0.0, 0.9],
                    [0.0, 0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.9, 0.0, 0.0, 0.1],
                ]
            )
        )

    # Negative value / more spin-up electrons than total electrons.
    with pytest.raises(ValueError):
        somo_from_orbdens(np.array([[0.0, 1.0, 0.0, -0.5], [1.0, 0.0, 0.0, 0.0]]))

    # Odd number of paired electrons.
    with pytest.raises(ValueError):
        somo_from_orbdens(
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.2, 0.5, 0.0, 0.3],
                    [0.3, 0.5, 0.0, 0.2],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            )
        )


def test_minimal_space_from_orbdens():
    for nmo in range(1, 11):
        for _ in range(1000):
            orbdens = np.random.random((nmo, 4))

            for row in orbdens:
                row /= np.sum(row)

            pairinfo = np.random.random((nmo, nmo))
            pa = PairInfoAnalyzer(pairinfo, orbdens)
            mo_list = pa.minimal_space()

            spin_per_mo = orbdens[:, 1] - orbdens[:, 2]
            total_spin = np.sum(spin_per_mo)
            space_spin = np.sum(spin_per_mo[mo_list])
            assert abs(total_spin - space_spin) < 0.5


def test_pairinfo_sum():
    # Test the computation of pair information for overall spaces of 1 to 19 orbitals.
    for nmo in range(1, 20):

        # Symmetric random matrix to fake pair information.
        pairinfo = np.random.random((nmo, nmo))
        pairinfo = pairinfo + pairinfo.T

        # Random matrix to fake the one-orbital density.
        orbdens = np.random.random((nmo, 4))

        pa = PairInfoAnalyzer(pairinfo, orbdens)

        # Construct random lists of varying length. Four copies of each list are created to test
        # that caching of results is correct.
        mo_lists = [random.sample(range(nmo), k=nsel) for _ in range(5) for nsel in range(nmo)] * 4

        # Arrange the lists in random order.
        random.shuffle(mo_lists)

        # Verify correctness for all list elements.
        for mo_list in mo_lists:
            assert isclose(
                pa.pairinfo_sum(mo_list),
                np.sum(pairinfo[mo_list, :][:, mo_list]),
                abs_tol=1e-12,
                rel_tol=0.0,
            )


def test_pairinfo_per_orbital():
    for nmo in range(10):

        # Symmetric random matrix to fake pair information.
        pairinfo = np.random.random((nmo, nmo))
        pairinfo = pairinfo + pairinfo.T

        # Random matrix to fake the one-orbital density.
        orbdens = np.random.random((nmo, 4))

        pa = PairInfoAnalyzer(pairinfo, orbdens)

        # Verify the shape and the sum over all elements are correct.
        pvec = pa.pairinfo_per_orbital()
        assert pvec.shape == (nmo,)
        assert isclose(np.sum(pvec), np.sum(pairinfo), abs_tol=1e-12, rel_tol=0.0)


def test_sanitize_active_spaces():
    # Orbital density with ten electrons. Net sum of unpaired electrons is 1 (= spin).
    orbdens = np.array(
        [
            [0.00, 0.00, 0.00, 1.00],
            [0.02, 0.00, 0.00, 0.98],
            [0.05, 0.10, 0.05, 0.80],
            [0.35, 0.55, 0.00, 0.10],
            [0.01, 0.49, 0.49, 0.01],
            [0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.55, 0.40, 0.00, 0.05],
            [0.98, 0.00, 0.00, 0.02],
            [1.00, 0.00, 0.00, 0.00],
        ]
    )

    # Add dummy information to convert MO lists info typed dicts for testing.
    def decorate_mo_list(mo_list: list[int], nel: int) -> MOListInfo:
        return MOListInfo(
            mo_list=mo_list,
            nel=nel,
            minimal=False,
            pairinfo_sum=0.0,
            max_increment=0.0,
            min_decrement=0.0,
            min_entropy=0.0,
            min_edge_sum=0.0,
        )

    # Create object with random pair info (would normally be a symmetric matrix).
    pa = PairInfoAnalyzer(pairinfo=np.random.random((10, 10)), orbdens=orbdens, spin=1)

    input_spaces: ActiveSpacesDict = {
        (1, 1): [decorate_mo_list([5], 1)],  # ok
        (0, 2): [decorate_mo_list([8, 9], 0)],  # fail: (0, N) active space
        (1, 2): [decorate_mo_list([3, 7], 1)],  # ok
        (6, 3): [decorate_mo_list([0, 1, 2], 6)],  # fail: (2N, N) active space
        (4, 4): [
            decorate_mo_list([0, 1, 8, 9], 4),  # fail: no unpaired electrons
            decorate_mo_list([1, 4, 5, 8], 4),  # fail: one unpaired electron out of
            # four electrons (odd / even parity)
        ],
        (5, 5): [
            decorate_mo_list([0, 1, 5, 8, 9], 5),  # ok
            decorate_mo_list([0, 1, 2, 8, 9], 5),  # fail: no unpaired electron
            decorate_mo_list([0, 1, 2, 8, 9], 6),  # fail: no unpaired electron
            decorate_mo_list([2, 3, 5, 6, 7], 5),  # ok
        ],
        (5, 6): [decorate_mo_list([2, 3, 4, 6, 7, 8], 5)],  # unpaired electrons: 0 instead of 1
        (9, 9): [decorate_mo_list([0, 1, 2, 3, 5, 6, 7, 8, 9], 9)],  # ok
    }

    # Perform the actual sanitization that will be tested.
    filtered_spaces = pa.sanitize_active_spaces(input_spaces)

    # A complete dictionary of spaces that should remain after sanitization.
    assert filtered_spaces == {
        (1, 1): [decorate_mo_list([5], 1)],
        (1, 2): [decorate_mo_list([3, 7], 1)],
        (5, 5): [decorate_mo_list([0, 1, 5, 8, 9], 5), decorate_mo_list([2, 3, 5, 6, 7], 5)],
        (9, 9): [decorate_mo_list([0, 1, 2, 3, 5, 6, 7, 8, 9], 9)],
    }


space_parameters = {
    "C2H4": {
        "orbdens": np.array(
            [
                [2.29677150e-04, 0.00000000e00, 0.00000000e00, 9.99770323e-01],
                [3.26958912e-02, 0.00000000e00, 0.00000000e00, 9.67304109e-01],
                [9.67173387e-01, 0.00000000e00, 0.00000000e00, 3.28266131e-02],
                [9.99901045e-01, 0.00000000e00, 0.00000000e00, 9.89551778e-05],
            ]
        ),
        "pairinfo": np.array(
            [
                [-4.59248796e-04, 1.92334923e-05, 5.80085500e-04, 3.19178599e-04],
                [1.92334923e-05, -6.32537397e-02, 1.26413832e-01, 7.44139943e-05],
                [5.80085500e-04, 1.26413832e-01, -6.34980532e-02, 2.18894937e-06],
                [3.19178599e-04, 7.44139943e-05, 2.18894937e-06, -1.97890771e-04],
            ]
        ),
        "spaces": {
            (2, 2): [
                MOListInfo(
                    mo_list=[1, 2],
                    nel=2,
                    minimal=False,
                    pairinfo_sum=0.12607587110000001,
                    max_increment=0.0007393891885999759,
                    min_decrement=0.18932961080000002,
                    min_entropy=0.14399195,
                    min_edge_sum=0.06325374,
                ),
                MOListInfo(
                    mo_list=[0, 3],
                    nel=2,
                    minimal=False,
                    pairinfo_sum=-1.8782369000000054e-05,
                    max_increment=-0.062333504301259994,
                    min_decrement=0.00017910840199999995,
                    min_entropy=0.0010114,
                    min_edge_sum=0.00019789,
                ),
                MOListInfo(
                    mo_list=[0, 2],
                    nel=2,
                    minimal=False,
                    pairinfo_sum=-0.062797130996,
                    max_increment=0.1896123912846,
                    min_decrement=-0.0623378822,
                    min_entropy=0.00215408,
                    min_edge_sum=0.00045925,
                ),
                # List [1, 3] is not part of the spaces is its pair information sum is less then the
                # "pair information" of [1] with itself.
            ],
            (2, 3): [
                MOListInfo(
                    mo_list=[1, 2, 3],
                    nel=2,
                    minimal=False,
                    pairinfo_sum=0.12603118621634,
                    max_increment=0.0013777463866000705,
                    min_decrement=-4.468488366002332e-05,
                    min_entropy=0.0010114,
                    min_edge_sum=0.00019789,
                ),
                # List [0, 2, 3] is not part of the spaces is its pair information sum is below that
                # of list [1, 2]
            ],
            (4, 3): [
                MOListInfo(
                    mo_list=[0, 1, 2],
                    nel=4,
                    minimal=False,
                    pairinfo_sum=0.12681526028860002,
                    max_increment=0.0005936723143400435,
                    min_decrement=0.0007393891886000037,
                    min_entropy=0.00215408,
                    min_edge_sum=0.00045925,
                ),
                # List [0, 1, 3] is not part of the spaces is its pair information sum is below that
                # of list [1, 2]
            ],
            (4, 4): [
                MOListInfo(
                    mo_list=[0, 1, 2, 3],
                    nel=4,
                    minimal=False,
                    pairinfo_sum=0.12740893260294006,
                    max_increment=0.0,
                    min_decrement=0.0005936723143400435,
                    min_entropy=0.0010114,
                    min_edge_sum=0.00019789,
                )
            ],
        },
    },
    "C3H5": {
        "orbdens": np.array(
            [
                [1.71464382e-02, 4.44827361e-02, 7.76122853e-03, 9.30609597e-01],
                [2.16493490e-15, 9.65206295e-01, 3.47937054e-02, 4.18190060e-29],
                [9.30609597e-01, 4.25549339e-02, 9.68903070e-03, 1.71464382e-02],
            ]
        ),
        "pairinfo": np.array(
            [
                [-3.12227985e-02, -4.44089210e-15, 9.24208698e-02],
                [-4.44089210e-15, 6.71662070e-02, -1.94289029e-16],
                [9.24208698e-02, -1.94289029e-16, -3.10886477e-02],
            ]
        ),
        "spaces": {
            (1, 1): [
                MOListInfo(
                    mo_list=[1],
                    nel=1,
                    minimal=True,
                    pairinfo_sum=6.71662070e-02,
                    max_increment=-3.108865e-02,
                    min_decrement=6.716621e-02,
                    min_entropy=0.15102961,
                    min_edge_sum=0.06716621,
                )
            ],
            (1, 2): [
                MOListInfo(
                    mo_list=[1, 2],
                    nel=1,
                    minimal=False,
                    pairinfo_sum=0.03607755929999962,
                    max_increment=0.1536189410999911,
                    min_decrement=-0.031088647700000388,
                    min_entropy=0.31591192,
                    min_edge_sum=0.06133222,
                )
            ],
            (3, 2): [
                MOListInfo(
                    mo_list=[0, 1],
                    nel=3,
                    minimal=False,
                    pairinfo_sum=0.03594340849999113,
                    max_increment=0.15375309189999958,
                    min_decrement=-0.031222798500008878,
                    min_entropy=0.31281019,
                    min_edge_sum=0.06119807,
                )
            ],
            (3, 3): [
                MOListInfo(
                    mo_list=[0, 1, 2],
                    nel=3,
                    minimal=False,
                    pairinfo_sum=0.18969650039999073,
                    max_increment=0.0,
                    min_decrement=0.1536189410999911,
                    min_entropy=0.31281019,
                    min_edge_sum=0.06119807,
                )
            ],
        },
    },
}


def compare_nested_dicts(arg1, arg2, atol=1e-12, rtol=2e-6):
    if isinstance(arg1, dict) or isinstance(arg2, dict):

        assert arg1.keys() == arg2.keys()

        for key in arg1.keys():
            val1 = arg1[key]
            val2 = arg2[key]
            compare_nested_dicts(val1, val2, atol, rtol)

    elif isinstance(arg1, list) or isinstance(arg2, list):

        assert len(arg1) == len(arg2)
        for item1, item2 in zip(arg1, arg2):
            compare_nested_dicts(item1, item2, atol, rtol)

    elif isinstance(arg1, float) or isinstance(arg2, float):

        assert isclose(arg1, arg2, abs_tol=atol, rel_tol=rtol)

    else:

        assert arg1 == arg2


def compare_active_space_dicts(d1: ActiveSpacesDict, d2: ActiveSpacesDict, atol=1e-8, rtol=2e-6):

    def is_close(a: MOListInfo, b: MOListInfo, atol, rtol) -> bool:
        return all(
            [
                a.mo_list == b.mo_list,
                a.nel == b.nel,
                a.minimal == b.minimal,
                isclose(a.pairinfo_sum, b.pairinfo_sum, abs_tol=atol, rel_tol=rtol),
                isclose(a.max_increment, b.max_increment, abs_tol=atol, rel_tol=rtol),
                isclose(a.min_decrement, b.min_decrement, abs_tol=atol, rel_tol=rtol),
                isclose(a.min_entropy, b.min_entropy, abs_tol=atol, rel_tol=rtol),
                isclose(a.min_edge_sum, b.min_edge_sum, abs_tol=atol, rel_tol=rtol),
            ]
        )

    assert d1.keys() == d2.keys()

    for space, d1_info_list in d1.items():
        assert len(d1_info_list) == len(d2[space])
        for i, mo_info in enumerate(d1_info_list):
            assert is_close(mo_info, d2[space][i], atol=atol, rtol=rtol) == True


@pytest.mark.parametrize("molecule", space_parameters.keys())
def test_all_candidate_spaces(molecule):
    parms = space_parameters[molecule]

    pa = PairInfoAnalyzer(orbdens=parms["orbdens"], pairinfo=parms["pairinfo"])
    spaces = pa.generate_candidate_spaces()
    compare_active_space_dicts(spaces, parms["spaces"], atol=1e-8)

    # also test mapping, with an mo list where initial_mos[i] == i we should get the same result
    initial_mos = np.arange(parms["orbdens"].shape[0]).tolist()
    pa = PairInfoAnalyzer(orbdens=parms["orbdens"], pairinfo=parms["pairinfo"])
    spaces = pa.generate_candidate_spaces(mo_list_full=initial_mos)
    compare_active_space_dicts(spaces, parms["spaces"], atol=1e-8)


def test_less_or_equal():
    assert less_or_equal(1.0, 2.0, rel_tol=0.1)
    assert not less_or_equal(2.0, 1.0, rel_tol=0.1)
    assert less_or_equal(1.01, 1.0, rel_tol=0.02, abs_tol=0.0)
    assert less_or_equal(1.01, 1.0, rel_tol=0.0, abs_tol=0.02)
    assert not less_or_equal(1.01, 1.0, rel_tol=1e-4, abs_tol=1e-4)
    assert less_or_equal(1.0, 1.01, rel_tol=0.0, abs_tol=0.0)
    assert less_or_equal(1e-10, 1e-12, rel_tol=1e-3)
    assert less_or_equal(-10.0, -9.9, rel_tol=1e-3)
    assert not less_or_equal(-9.9, -10.0, rel_tol=1e-3)
    assert less_or_equal(-9.9, -10.0, rel_tol=2e-2)

    with pytest.raises(ValueError):
        less_or_equal(1.0, 1.0, rel_tol=-0.01)
    with pytest.raises(ValueError):
        less_or_equal(1.0, 1.0, rel_tol=0.01, abs_tol=-0.01)
