# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import numpy as np
import pytest
from pyscf import scf

from asf import wrapper
from asf.asfbase import merge_active_spaces
from asf.preselection import MP2NatorbPreselection

from .fixtures.molecules import create_mol


def test_merge_active_spaces_closedshell():
    nitrogen = create_mol("nitrogen")
    same_space = ((6, [4, 5, 6, 7, 8, 9]),) * 3
    subset = (same_space[0], (4, [5, 6, 7, 8]), same_space[0])
    complement = ((2, [6, 7]), (4, [4, 5, 8, 9]), same_space[0])
    overlapping = (
        (6, [4, 5, 6, 7, 8, 9, 10, 11]),
        (8, [2, 3, 5, 6, 7, 8]),
        (10, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    )
    right_empty = ((2, [6, 7]), (0, []), (2, [6, 7]))
    left_empty = ((0, []), (4, [4, 5, 8, 9]), (4, [4, 5, 8, 9]))
    all_empty = ((0, []), (0, []), (0, []))
    test_spaces = [same_space, subset, complement, overlapping, right_empty, left_empty, all_empty]

    for (nel1, mo_list1), (nel2, mo_list2), (nel_ref, mo_list_ref) in test_spaces:
        nel, mo_list = merge_active_spaces(nitrogen, nel1, mo_list1, nel2, mo_list2)
        assert nel == nel_ref
        assert mo_list == mo_list_ref


def test_merge_active_spaces_openshell():
    mol = create_mol("OH_radical")

    same_space = ((7, [1, 2, 3, 4, 5]),) * 3
    subset = ((5, [2, 3, 4, 5]), same_space[0], same_space[0])
    subset_small = ((1, [4]), (3, [3, 4, 5]), (3, [3, 4, 5]))
    overlapping = ((5, [2, 3, 4]), (3, [3, 4, 5, 6]), (5, [2, 3, 4, 5, 6]))
    test_spaces = [same_space, subset, subset_small, overlapping]

    for (nel1, mo_list1), (nel2, mo_list2), (nel_ref, mo_list_ref) in test_spaces:
        nel, mo_list = merge_active_spaces(mol, nel1, mo_list1, nel2, mo_list2)
        assert nel == nel_ref
        assert mo_list == mo_list_ref


def test_merge_active_spaces_exceptions():
    nitrogen = create_mol("nitrogen")
    bad_spaces = [
        ((5, [4, 5, 6, 7, 8, 9]), (6, [4, 5, 6, 7, 8, 9])),
        ((6, [4, 5, 6, 7, 8, 9]), (4, [4, 5, 6, 7, 8, 9])),
        ((2, [6, 7]), (4, [8, 9, 10, 11])),
        ((2, []), (2, [5, 7])),
        ((0, []), (2, [])),
    ]

    for (nel1, mo_list1), (nel2, mo_list2) in bad_spaces:
        with pytest.raises(ValueError):
            merge_active_spaces(nitrogen, nel1, mo_list1, nel2, mo_list2)


def test_reorder_mos(nitrogen_RHF):
    mf = nitrogen_RHF
    nitrogen = nitrogen_RHF.mol
    mo_list = [2, 5, 6, 7, 8, 9, 10, 11]

    a, b = wrapper.reorder_mos(nitrogen, 6, mo_list, mf.mo_coeff)
    assert a == [4, 5, 6, 7, 8, 9, 10, 11]
    assert np.array_equal(b[:, 4], mf.mo_coeff[:, 2])


def test_loghead():
    wrapper.loghead("Calculating MP2 natural orbitals", verbose=True)


def test_sized_space_from_scf(nitrogen_RHF):
    space = wrapper.sized_space_from_scf(nitrogen_RHF, size=(4, 4))
    assert space.nel == 4
    assert space.mo_list == [5, 6, 7, 8]


def test_sized_space_from_scf_excited(nitrogen_RHF):
    space = wrapper.sized_space_from_scf(nitrogen_RHF, size=(4, 4), state=(0, 1))
    assert space.nel == 4
    assert space.mo_list == [5, 6, 7, 8]


def test_sized_space_from_scf_triplet(nitrogen_RHF):
    space = wrapper.sized_space_from_scf(nitrogen_RHF, size=(4, 4), state=(2, 0))
    assert space.nel == 4
    assert space.mo_list == [5, 6, 7, 8]


def test_sized_space_from_scf_openshell():
    mol = create_mol("nitric_oxide")
    mf = scf.UHF(mol)
    mf.kernel()

    space = wrapper.sized_space_from_scf(mf, size=(7, 6))
    assert space.nel == 7
    assert space.mo_list == [4, 5, 6, 7, 8, 9]


def test_create_asf_switched():
    nitrogen = create_mol("nitrogen")
    mf = scf.UHF(nitrogen)
    mf.kernel()
    dmrg_settings = {"maxM": 151}

    pre = MP2NatorbPreselection(mf)
    space = pre.select()
    # For DMRG
    dmrgci = wrapper.create_asf_switched_cisolver(
        nitrogen, initial_space=space, spin=2, switch_dmrg=4, dmrg_kwargs=dmrg_settings
    )
    dmrgci.calculate()
    # check spins
    assert dmrgci.casci.nelecas[0] - dmrgci.casci.nelecas[1] == 2

    casci = wrapper.create_asf_switched_cisolver(
        nitrogen, initial_space=space, spin=2, switch_dmrg=12
    )
    casci.calculate()

    # nelecas does not change in casci calculations even by asking for different spin
    assert casci.casci.fcisolver.spin == 2
    # Compare CASCI DMRGCI energies (DMRG converged to 1e-6 tolerance by default)
    assert abs(dmrgci.casci.e_tot - casci.casci.e_tot) < 1e-5


def test_sized_space_from_mol():
    nitrogen = create_mol("nitrogen")
    space = wrapper.sized_space_from_mol(nitrogen, size=(4, 4))
    assert space.nel == 4
    assert space.mo_list == [5, 6, 7, 8]


def test_find_from_scf(formaldehyde_UHF):
    space = wrapper.find_from_scf(
        formaldehyde_UHF, min_norb=4, max_norb=12, entropy_threshold=0.01
    )

    assert space.norb >= 4
    assert space.norb <= 12


def test_find_from_scf_excited(nitrogen_RHF):
    space = wrapper.find_from_scf(nitrogen_RHF, max_norb=10, states=[(2, 2)])
    assert space.norb <= 10
    assert space.mo_list == [5, 6, 7, 8]


def test_find_from_scf_multispin(nitrogen_RHF):
    space = wrapper.find_from_scf(nitrogen_RHF, max_norb=10, states=[(0, 1), (2, 1)])
    assert space.norb <= 10
    assert space.mo_list == [5, 6, 7, 8]


def test_find_from_mol():
    formaldehyde = create_mol("formaldehyde", spin=2)
    space = wrapper.find_from_mol(formaldehyde, min_norb=4, max_norb=12, entropy_threshold=0.01)

    assert space.norb >= 4
    assert space.norb <= 12
