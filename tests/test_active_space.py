# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import numpy as np
import pytest
from pyscf import gto

from asf.asfbase import ActiveSpace


@pytest.fixture(
    params=[
        (4, [4, 6, 8, 10], np.random.random((11, 11))),
        (6, [1, 3, 5, 7, 8, 9], np.random.random((10, 10))),
    ]
)
def space_dict(request):
    return {"nel": request.param[0], "mo_list": request.param[1], "mo_coeff": request.param[2]}


@pytest.fixture()
def formaldehyde():
    """Returns Formaldehyde as pyscf Mole object.

    Returns:
        mol:    Formaldehyde as pyscf Mole object.
    """
    mol = gto.Mole()
    mol.atom = """
    C 0.000 0.000 -0.533
    O 0.000 0.000 0.680
    H 0.000 -0.937 -1.118
    H 0.000 0.937 -1.118
    """
    mol.spin = 0
    mol.charge = 0
    mol.basis = "minao"
    mol.symmetry = False
    mol.build()
    return mol


def test_from_dict(space_dict):
    act = ActiveSpace.from_dict(space_dict)
    assert act.nel == space_dict["nel"]
    assert act.mo_list == space_dict["mo_list"]
    assert act.norb == len(space_dict["mo_list"])


def test_to_dict(space_dict):
    act = ActiveSpace(**space_dict)
    assert act.to_dict() == space_dict


def test_inconsistent_input():
    with pytest.raises(ValueError):
        ActiveSpace(nel=-1, mo_list=[1, 2, 3], mo_coeff=np.random.random((4, 4)))

    with pytest.raises(ValueError):
        ActiveSpace(nel=4, mo_list=[1, -2, 3], mo_coeff=np.random.random((4, 4)))

    with pytest.raises(ValueError):
        ActiveSpace(nel=4, mo_list=[1, 2, 3, 4], mo_coeff=np.random.random((2, 3)))


def test_merge_with(formaldehyde):
    dummy_mo_coeff = np.random.random((10, 10))
    act1 = ActiveSpace(nel=2, mo_list=[7, 8], mo_coeff=dummy_mo_coeff)
    act2 = ActiveSpace(nel=2, mo_list=[5, 9], mo_coeff=dummy_mo_coeff)
    merged = act1.merge_with(mol=formaldehyde, other=act2)
    assert merged.nel == 4
    assert merged.norb == 4
    assert merged.mo_list == [5, 7, 8, 9]


def test_instantiation_np_array(space_dict):
    mo_array = np.array(space_dict["mo_list"])
    act = ActiveSpace(nel=space_dict["nel"], mo_list=mo_array, mo_coeff=space_dict["mo_coeff"])
    assert type(act.mo_list) == list
    assert type(act.mo_list[0]) == int
