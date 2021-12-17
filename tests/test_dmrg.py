import math
import numpy

from pyscf import gto, scf, mcscf

from asf import ASFDMRG


def test__calc_rdm1s_Be():
    mol = gto.Mole(atom='Be 0 0 0', basis='6-31G', spin=0).build()
    mf = scf.RHF(mol).run()
    sf = ASFDMRG(mol, mf.mo_coeff, 4, 5)
    sf.runDMRG(maxM=1000, tol=1.0e-9)
    rdm1a, rdm1b = sf._calc_rdm1s()
    casci = mcscf.CASCI(mf, 5, 4).run()
    rdm1ref = casci.fcisolver.make_rdm1(casci.ci, 5, (2, 2))
    assert numpy.allclose(rdm1a, 0.5 * rdm1ref, atol=1.0e-5, rtol=0.0)
    assert numpy.allclose(rdm1b, 0.5 * rdm1ref, atol=1.0e-5, rtol=0.0)


def test__calc_rdm1s_Li():
    mol = gto.Mole(atom='Li 0 0 0', basis='6-31G', spin=1).build()
    mf = scf.ROHF(mol).run()
    sf = ASFDMRG(mol, mf.mo_coeff, 3, 5)
    sf.runDMRG(maxM=1000, tol=1.0e-9)
    rdm1a, rdm1b = sf._calc_rdm1s()
    casci = mcscf.CASCI(mf, 5, 3).run()
    rdm1a_ci, rdm1b_ci = casci.fcisolver.make_rdm1s(casci.ci, 5, (2, 1))
    assert numpy.allclose(rdm1a, rdm1a_ci, atol=1.0e-5, rtol=0.0)
    assert numpy.allclose(rdm1b, rdm1b_ci, atol=1.0e-5, rtol=0.0)


def test_getOrbDens_C():
    mol = gto.Mole(atom='C 0 0 0', basis='STO-3G', spin=2).build()
    mf = scf.ROHF(mol).run()
    sf = ASFDMRG(mol, mf.mo_coeff, 6, 5)
    sf.runDMRG(nroots=3, maxM=1000, tol=1.0e-9)
    orbdens = numpy.array([sf.getOrbDens(i) for i in range(3)])
    for i in range(3):

        # General checks.
        assert numpy.all(orbdens >= 0.0)
        assert numpy.allclose(numpy.sum(orbdens, axis=2), numpy.ones((3, 5)), atol=1.0e-8, rtol=0.0)

        # Checking that doubly occupied orbitals are mostly doubly occupied.
        assert numpy.allclose(orbdens[i, 0, :], [0.000, 0.000, 0.000, 1.000], atol=1e-2, rtol=0.0)
        assert numpy.allclose(orbdens[i, 1, :], [0.023, 0.000, 0.000, 0.977], atol=1e-2, rtol=0.0)

        # Three ways to put two unpaired electrons into three (2p) orbitals.
        # Sum of occupations of the three 2p orbitals equals two?
        assert math.isclose(numpy.sum(orbdens[i, 2:, 1]), 2.0, abs_tol=1e-4, rel_tol=0.0)
        # Sum of occupations of each 2p orbital over all three roots equals two?
        assert math.isclose(numpy.sum(orbdens[:, 2+i, 1]), 2.0, abs_tol=1e-4, rel_tol=0.0)
