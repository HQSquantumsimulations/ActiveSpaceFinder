# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
from math import isclose

import numpy as np

from asf import natorbs


def test_natural_spin_orbitals(OH_radical_UMP2):
    """
    Verifying the computation of natural spin orbitals.
    """
    mf = OH_radical_UMP2._scf  # no public getter for SCF object
    rdm1_mp2 = OH_radical_UMP2.make_rdm1(ao_repr=False)

    # Calculate natural spin orbitals
    mo_coeff = mf.mo_coeff
    S = mf.get_ovlp()
    nsocc, nsorb = natorbs.natural_spin_orbitals(rdm1_mp2, mo_coeff)

    # Eigenvalues must sum to the respective numbers of spin-up and spin-down orbitals.
    assert isclose(sum(nsocc[0]), 5.0, abs_tol=1e-12, rel_tol=0.0)
    assert isclose(sum(nsocc[1]), 4.0, abs_tol=1e-12, rel_tol=0.0)

    # Natural spin occupation numbers in descending order.
    assert np.array_equal(np.sort(nsocc[0]), nsocc[0, -1::-1])
    assert np.array_equal(np.sort(nsocc[1]), nsocc[1, -1::-1])

    # Checking explicitly the natural spin orbitals diagonalize the RDM spin components.
    for s in (0, 1):
        U = np.linalg.multi_dot([mo_coeff[s].T, S, nsorb[s]])
        rdm1_transformed = np.linalg.multi_dot([U.T, rdm1_mp2[s], U])
        assert np.allclose(rdm1_transformed, np.diag(nsocc[s]), atol=1e-12, rtol=0.0)
