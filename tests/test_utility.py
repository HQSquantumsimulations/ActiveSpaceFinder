import os
import math
import numpy

from pyscf import gto, scf, fci

from asf import utility


def get_mol(spin=0, charge=0, basis='minao', sym=False):
    """Returns Formaldehyde as pyscf Mole object.

    Returns:
        mol:    Formaldehyde as pyscf Mole object.
    """
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = """
    C 0.000 0.000 -0.533
    O 0.000 0.000 0.680
    H 0.000 -0.937 -1.118
    H 0.000 0.937 -1.118
    """
    mol.spin = spin
    mol.charge = charge
    mol.basis = basis
    mol.symmetry = sym
    mol.build()
    return mol


def test_tempname():
    with utility.tempname() as filename:
        f = open(filename, 'w')
        f.write('test\n')
        f.close()
        f = open(filename, 'r')
        lines = f.readlines()
        assert lines == ['test\n']
    assert not os.path.exists(filename)


# Utility functions
def test_pictures_Jmol():
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.kernel()
    mos = [2, 3, 4]
    utility.pictures_Jmol(mf.mol, mf.mo_coeff, mos)
    files_exist = True
    for filename in ['mo_2.png', 'mo_3.png', 'mo_4.png']:
        if os.path.exists(filename):
            os.remove(filename)
        else:
            files_exist = False
    assert files_exist


def test_rdm1s_from_rdm12_1e():
    # single electron
    N = 1
    S = 0.5
    rdm1 = numpy.array([[1.0]])
    rdm2 = numpy.array([[[[0.0]]]])
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(N, S, rdm1, rdm2)
    assert numpy.allclose(rdm1a, 1.0*rdm1, atol=1.0e-12, rtol=0.0)
    assert numpy.allclose(rdm1b, 0.0*rdm1, atol=1.0e-12, rtol=0.0)


def test_rdm1s_from_rdm12_2e_singlet():
    # two electrons, singlet
    N = 2
    S = 0.0
    rdm1 = numpy.array([[2.0]])
    rdm2 = numpy.array([[[[2.0]]]])
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(N, S, rdm1, rdm2)
    assert numpy.allclose(rdm1a, 0.5*rdm1, atol=1.0e-12, rtol=0.0)
    assert numpy.allclose(rdm1b, 0.5*rdm1, atol=1.0e-12, rtol=0.0)


def test_rdm1s_from_rdm12_He():
    mol = gto.Mole(atom='He 0 0 0', basis='6-31G', spin=0).build()
    mf = scf.RHF(mol).run()
    fcisolver = fci.FCI(mf).run()
    assert mf.mo_coeff.shape[1] == 2
    rdm1, rdm2 = fcisolver.make_rdm12(fcisolver.ci, 2, (1, 1))
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(2, 0.0, rdm1, rdm2)
    assert numpy.allclose(rdm1a, 0.5 * rdm1, atol=1.0e-8, rtol=0.0)
    assert numpy.allclose(rdm1b, 0.5 * rdm1, atol=1.0e-8, rtol=0.0)


def test_rdm1s_from_rdm12_Li():
    mol = gto.Mole(atom='Li 0 0 0', basis='6-31G', spin=1).build()
    mf = scf.ROHF(mol).run()
    fcisolver = fci.FCI(mf).run()
    assert mf.mo_coeff.shape[1] == 9
    rdm1, rdm2 = fcisolver.make_rdm12(fcisolver.ci, 9, (2, 1))
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(3, 0.5, rdm1, rdm2)
    assert numpy.allclose(rdm1a + rdm1b, rdm1, atol=1.0e-8, rtol=0.0)
    assert math.isclose(numpy.trace(rdm1a - rdm1b), 1.0, abs_tol=1.0e-8, rel_tol=0.0)


def test_rdm1s_from_rdm12_C():
    mol = gto.Mole(atom='C 0 0 0', basis='6-31G', spin=2).build()
    mf = scf.ROHF(mol).run()
    fcisolver = fci.FCI(mf).run()
    assert mf.mo_coeff.shape[1] == 9
    rdm1, rdm2 = fcisolver.make_rdm12(fcisolver.ci, 9, (4, 2))
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(6, 1.0, rdm1, rdm2)
    assert numpy.allclose(rdm1a + rdm1b, rdm1, atol=1.0e-8, rtol=0.0)
    assert math.isclose(numpy.trace(rdm1a - rdm1b), 2.0, abs_tol=1.0e-8, rel_tol=0.0)


def test_rdm1s_from_rdm12_N():
    mol = gto.Mole(atom='N 0 0 0', basis='6-31G', spin=3).build()
    mf = scf.ROHF(mol).run()
    fcisolver = fci.FCI(mf).run()
    assert mf.mo_coeff.shape[1] == 9
    rdm1, rdm2 = fcisolver.make_rdm12(fcisolver.ci, 9, (5, 2))
    rdm1a, rdm1b = utility.rdm1s_from_rdm12(7, 1.5, rdm1, rdm2)
    assert numpy.allclose(rdm1a + rdm1b, rdm1, atol=1.0e-8, rtol=0.0)
    assert math.isclose(numpy.trace(rdm1a - rdm1b), 3.0, abs_tol=1.0e-8, rel_tol=0.0)
