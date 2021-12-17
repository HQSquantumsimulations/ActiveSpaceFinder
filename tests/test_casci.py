import math
import numpy
from pyscf import gto, scf

from asf import ASFCI

def test_getOrbDens_C():
    mol = gto.Mole(atom='C 0 0 0', basis='STO-3G', spin=2).build()
    mf = scf.ROHF(mol).run()
    sf = ASFCI(mol, mf.mo_coeff, 6, 5)
    sf.runCI(nroots=3)
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
