# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import math
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from pyscf import fci, gto, scf

from asf import utility

from .fixtures.molecules import create_mol


def test_tempname():
    with utility.tempname() as filename:
        f = open(filename, "w")
        f.write("test\n")
        f.close()
        f = open(filename, "r")
        lines = f.readlines()
        assert lines == ["test\n"]
    assert not os.path.exists(filename)


@pytest.mark.parametrize("mo_filename", ["/path/to/mo.molden", Path("/path/to/mo.molden")])
def test_write_jmol_script(mo_filename):
    stream = StringIO()
    utility.write_jmol_script(stream=stream, mo_list=[2, 3, 7], mo_filename=mo_filename)
    script = stream.getvalue()
    ref_script = """load "/path/to/mo.molden"
background white
frank off
spacefill off
wireframe 0.1
mo fill nomesh translucent
mo color yellow purple
mo resolution 20
mo cutoff 0.040000
mo titleformat "MO %I of %N"
rotate 0.000000 x
rotate 0.000000 y
rotate 0.000000 z
zoom 60
mo 3
write image png "mo_2.png"
mo 4
write image png "mo_3.png"
mo 8
write image png "mo_7.png"
"""
    assert script == ref_script


def test_rdm1s_from_rdm12_1e():
    # single electron
    N = 1
    S = 0.5
    rdm1 = np.array([[1.0]])
    rdm2 = np.array([[[[0.0]]]])
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(N, S, rdm1, rdm2)
    assert np.allclose(rdm1a, 1.0 * rdm1, atol=1.0e-12, rtol=0.0)
    assert np.allclose(rdm1b, 0.0 * rdm1, atol=1.0e-12, rtol=0.0)


def test_rdm1s_from_rdm12_2e_singlet():
    # two electrons, singlet
    N = 2
    S = 0.0
    rdm1 = np.array([[2.0]])
    rdm2 = np.array([[[[2.0]]]])
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(N, S, rdm1, rdm2)
    assert np.allclose(rdm1a, 0.5 * rdm1, atol=1.0e-12, rtol=0.0)
    assert np.allclose(rdm1b, 0.5 * rdm1, atol=1.0e-12, rtol=0.0)


def test_rdm1s_from_rdm12_He():
    mol = create_mol("helium_atom")
    mf = scf.RHF(mol).run()
    fcisolver = fci.FCI(mf).run()
    assert mf.mo_coeff.shape[1] == 2
    rdm1, rdm2 = fcisolver.make_rdm12(fcisolver.ci, 2, (1, 1))
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(2, 0.0, rdm1, rdm2)
    assert np.allclose(rdm1a, 0.5 * rdm1, atol=1.0e-8, rtol=0.0)
    assert np.allclose(rdm1b, 0.5 * rdm1, atol=1.0e-8, rtol=0.0)


@pytest.mark.parametrize(
    "data",
    [
        {"mol": create_mol("lithium_atom"), "norb": 9},
        {"mol": create_mol("carbon_atom"), "norb": 9},
        {"mol": create_mol("nitrogen_atom"), "norb": 9},
    ],
)
def test_rdm1s_from_rdm12_openshell(data):
    mol = data["mol"]
    mf = scf.ROHF(mol).run()
    fcisolver = fci.FCI(mf).run()
    assert mf.mo_coeff.shape[1] == data["norb"]
    rdm1, rdm2 = fcisolver.make_rdm12(fcisolver.ci, data["norb"], mol.nelec)
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(mol.nelectron, mol.spin / 2, rdm1, rdm2)
    assert np.allclose(rdm1a + rdm1b, rdm1, atol=1.0e-8, rtol=0.0)
    assert math.isclose(np.trace(rdm1a - rdm1b), mol.spin, abs_tol=1.0e-8, rel_tol=0.0)
