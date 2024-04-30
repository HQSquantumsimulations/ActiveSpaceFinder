# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import numpy as np
from pyscf import gto, mcscf, scf

from asf import ASFDMRG

from .fixtures.molecules import create_mol


def test__calc_rdm1s_Be():
    mol = create_mol("beryllium_atom")
    mf = scf.RHF(mol).run()
    sf = ASFDMRG(mol, mf.mo_coeff, 4, 5, maxM=1000, tol=1.0e-9, nroots=2)
    # Additional root to ensure convergence to the ground state.
    sf.calculate()
    rdm1a, rdm1b = sf._calc_rdm1s()
    casci = mcscf.CASCI(mf, 5, 4).run()
    rdm1ref = casci.fcisolver.make_rdm1(casci.ci, 5, (2, 2))
    assert np.allclose(rdm1a, 0.5 * rdm1ref, atol=1.0e-5, rtol=0.0)
    assert np.allclose(rdm1b, 0.5 * rdm1ref, atol=1.0e-5, rtol=0.0)


def test__calc_rdm1s_Li():
    mol = create_mol("lithium_atom")
    mf = scf.ROHF(mol).run()
    sf = ASFDMRG(mol, mf.mo_coeff, 3, 5, maxM=1000, tol=1.0e-9, nroots=2)
    # Additional root to ensure convergence to the ground state.
    sf.calculate()
    rdm1a, rdm1b = sf._calc_rdm1s()
    casci = mcscf.CASCI(mf, 5, 3).run()
    rdm1a_ci, rdm1b_ci = casci.fcisolver.make_rdm1s(casci.ci, 5, (2, 1))
    assert np.allclose(rdm1a, rdm1a_ci, atol=1.0e-5, rtol=0.0)
    assert np.allclose(rdm1b, rdm1b_ci, atol=1.0e-5, rtol=0.0)


def test_one_orbital_density_C():
    # System setup
    mol = create_mol("carbon_atom", basis="STO-3G")
    mf = scf.ROHF(mol).run()

    # Settings similar to https://block2.readthedocs.io/en/latest/user/dmrg-scf.html#dmrg-ic-nevpt2
    # Recommended by block2 developer: https://github.com/block-hczhai/block2-preview/issues/63
    block2_settings = dict(
        nroots=3,
        scheduleSweeps=[0, 4, 8, 12, 16],
        scheduleMaxMs=[250, 500, 500, 500, 500],
        scheduleTols=[1e-8, 1e-10, 1e-12, 1e-12, 1e-12],
        scheduleNoises=[1e-4, 1e-4, 5e-5, 5e-5, 0.0],
        maxIter=30,
        twodot_to_onedot=20,
        block_extra_keyword=[
            "singlet_embedding",
            "full_fci_space",
            "fp_cps_cutoff 0",
            "cutoff 0",
        ],
        tol=1e-14,
    )
    sf = ASFDMRG(mol, mf.mo_coeff, 6, 5, fcisolver_kwargs=block2_settings)
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
