# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import numpy as np
from pyscf import scf

from asf import ASFCI, preselection

from .fixtures.molecules import create_mol


def test_one_orbital_density_C():
    # System setup and CI calculation
    mol = create_mol("carbon_atom", basis="STO-3G")
    mf = scf.ROHF(mol).run()
    sf = ASFCI(mol, mf.mo_coeff, 6, 5, nroots=3)
    sf.calculate()

    # One-orbital densities for each root.
    orbdens_roots = np.array([sf.one_orbital_density(i) for i in range(3)])

    # One-orbital density must be non-negative (numerically).
    assert np.all(orbdens_roots >= -1e-14)

    # Sum of contributions for each orbital in each state equals one.
    assert np.allclose(np.sum(orbdens_roots, axis=2), np.ones((3, 5)), atol=1.0e-8, rtol=0.0)

    for i in range(3):
        # Checking occupations of the 1s and 2s orbitals.
        assert np.allclose(
            orbdens_roots[i, 0, :], [0.000, 0.000, 0.000, 1.000], atol=1e-2, rtol=0.0
        )
        assert np.allclose(
            orbdens_roots[i, 1, :], [0.023, 0.000, 0.000, 0.977], atol=1e-2, rtol=0.0
        )

    # Number of unpaired electrons per root and orbital.
    single_occupations = orbdens_roots[:, :, 1] - orbdens_roots[:, :, 2]

    # There need to be two unpaired electrons per state.
    assert np.allclose(np.sum(single_occupations, axis=1), 2.0, atol=1e-4, rtol=0.0)

    # Summed over all three states, there are two spin-up electrons per 2p-orbital.
    assert np.allclose(np.sum(single_occupations[:, 2:], axis=0), 2.0, atol=1e-4, rtol=0.0)


def test_init_from_active_space(nitrogen_RHF):
    space = preselection.MP2NatorbPreselection(nitrogen_RHF).select()
    sf = ASFCI.from_active_space(nitrogen_RHF.mol, space, nroots=2, spin_shift=0.2)

    assert sf.nel == 10
    assert np.array_equal(sf.mo_list, [i for i in range(2, 12)])
    assert sf.fcisolver_kwargs["nroots"] == 2
    assert sf.spin_shift == 0.2
