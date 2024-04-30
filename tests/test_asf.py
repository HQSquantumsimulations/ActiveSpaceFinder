# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import os

import numpy as np
import pytest
from pyscf import gto, lo, scf, symm

import asf

from .fixtures.molecules import create_mol

NATOCC_TOL = 1e-7
""" Tolerance for total number of electrons from natural occupation numbers."""


def test_setMOList(formaldehyde_RHF):
    asfci = asf.ASFCI(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, nel=4, mo_list=[6, 7, 8])
    assert asfci.norb == 3


def test_runCI(formaldehyde_RHF):
    asfci = asf.ASFCI(
        formaldehyde_RHF.mol,
        formaldehyde_RHF.mo_coeff,
        nel=4,
        mo_list=[6, 7, 8],
        fcisolver_kwargs=dict(conv_tol=1e-11),
    )
    asfci.calculate()
    assert abs(asfci.casci.e_tot + 113.69506678408428) < 1e-9
    assert asfci.casci.fcisolver.conv_tol == 1e-11


def test_one_orbital_density(formaldehyde_RHF):
    asfci = asf.ASFCI(
        formaldehyde_RHF.mol,
        formaldehyde_RHF.mo_coeff,
        nel=4,
        mo_list=[6, 7, 8, 9],
        fcisolver_kwargs=dict(conv_tol=1e-11),
    )
    asfci.calculate()
    orbd = np.array(asfci.one_orbital_density())

    orbd_o = np.array(
        [
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            [5.66923410e-02, 1.34973881e-04, 1.34973881e-04, 9.43037711e-01],
            [1.69640949e-03, 0.00000000e00, 0.00000000e00, 9.98303591e-01],
            [9.42286343e-01, 1.34973881e-04, 1.34973881e-04, 5.74437093e-02],
            [9.99054959e-01, 0.00000000e00, 0.00000000e00, 9.45041172e-04],
            [1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        ]
    )
    for x, i in enumerate(orbd_o):
        assert np.allclose(orbd[x], i, atol=asfci.casci.fcisolver.conv_tol)
    assert orbd.shape == (12, 4)


def test_one_orbital_entropy(formaldehyde_RHF):
    asfci = asf.ASFCI(
        formaldehyde_RHF.mol,
        formaldehyde_RHF.mo_coeff,
        nel=4,
        mo_list=[6, 7, 8, 9],
        fcisolver_kwargs=dict(conv_tol=1e-11),
    )
    asfci.calculate()
    S1 = asfci.one_orbital_entropy()
    S1_o = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22042718, 0.01251678, 0.22253437, 0.00752613, 0.0, 0.0]
    assert np.allclose(S1, S1_o, atol=asfci.casci.fcisolver.conv_tol)


def test_runDMRG(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    asfdmrg = asf.ASFDMRG(
        formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, 4, 4, maxM=150, tol=1e-10
    )
    asfdmrg.calculate()
    assert abs(asfdmrg.casci.e_tot + 113.69630979261541) < 1e-7
    assert asfdmrg.casci.fcisolver.tol == 1e-10


def test_one_orbital_density_dmrg(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    asfdmrg = asf.ASFDMRG(
        formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, 4, 4, maxM=150, tol=1e-10
    )
    asfdmrg.calculate()
    orbd = asfdmrg.one_orbital_density()
    orbd_o = [
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        [5.66923951e-02, 1.34971806e-04, 1.34971806e-04, 9.43037661e-01],
        [1.69640193e-03, 1.21014310e-14, 1.09912079e-14, 9.98303598e-01],
        [9.42286300e-01, 1.34971806e-04, 1.34971806e-04, 5.74437561e-02],
        [9.99054959e-01, 1.12169388e-14, 1.12165052e-14, 9.45040944e-04],
        [1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
    ]
    for x, i in enumerate(orbd_o):
        assert np.allclose(orbd[x], i, atol=1e-7)
    assert orbd.shape == (12, 4)


def test_entropy_selection_nocumulant(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    sf = asf.ASFCI(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff)
    sf.calculate()
    space = sf.entropy_selection(useCumulant=False)
    assert space.nel == 2
    assert space.mo_list == [6, 8]


def test_entropy_selection_yescumulant(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    sf = asf.ASFCI(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff)
    sf.calculate()
    space = sf.entropy_selection()
    assert space.nel == 2
    assert space.mo_list == [6, 8]


def test_entropy_selection_roots_CI_nocumulant(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    sf = asf.ASFCI(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, nroots=2)
    sf.calculate()
    space_0 = sf.entropy_selection(root=0, useCumulant=False)
    assert space_0.nel == 2
    assert space_0.mo_list == [6, 8]
    # entropy of this root
    # entropy_ref = [5.80858656e-05, 2.75277339e-04, 5.36580738e-02, 8.86412207e-02,
    #                8.20611595e-02, 9.89239187e-02, 2.67710551e-01, 1.01588892e-01,
    #                2.75214653e-01, 1.13480251e-01, 1.31990272e-01, 1.11464837e-01]
    # assert np.allclose(sf.entropy, entropy_ref, atol=1e-7)
    space_1 = sf.entropy_selection(root=1, useCumulant=False)
    assert space_1.nel == 6
    assert space_1.mo_list == [4, 6, 7, 8, 9, 10]
    # entropy_ref = [4.29475301e-05, 2.84879828e-04, 5.31488611e-02, 1.14860646e-01,
    #                3.99563236e-01, 9.55310707e-02, 3.16982530e-01, 9.57102030e-01,
    #                9.06211823e-01, 1.40129203e-01, 1.36965856e-01, 1.04377069e-01]
    # assert np.allclose(sf.entropy, entropy_ref, atol=1e-7)


def test_entropy_selection_roots_CI_yescumulant(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    sf = asf.ASFCI(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, nroots=2)
    sf.calculate()
    space_0 = sf.entropy_selection(root=0)
    assert space_0.nel == 2
    assert space_0.mo_list == [6, 8]
    space_1 = sf.entropy_selection(root=1)
    assert space_1.nel == 8
    assert space_1.mo_list == [3, 4, 6, 7, 8, 9, 10]


def test_entropy_selection_roots_DMRG_nocumulant(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    sf = asf.ASFDMRG(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, nroots=2)
    sf.calculate()
    space_0 = sf.entropy_selection(root=0, useCumulant=False)
    assert space_0.nel == 2
    assert space_0.mo_list == [6, 8]
    space_1 = sf.entropy_selection(root=1, useCumulant=False)
    assert space_1.nel == 6
    assert space_1.mo_list == [4, 6, 7, 8, 9, 10]


def test_entropy_selection_roots_DMRG_yescumulant(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    sf = asf.ASFDMRG(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, nroots=2)
    sf.calculate()
    space_0 = sf.entropy_selection(root=0)
    assert space_0.nel == 2
    assert space_0.mo_list == [6, 8]
    space_1 = sf.entropy_selection(root=1)
    assert space_1.nel == 8
    assert space_1.mo_list == [3, 4, 6, 7, 8, 9, 10]


def test_restricted_natural_orbitals(tmpdir, formaldehyde_MP2):
    os.chdir(tmpdir)
    mf = formaldehyde_MP2._scf  # no public getter for SCF object
    pt = formaldehyde_MP2
    occ, no = asf.natorbs.restricted_natural_orbitals(pt.make_rdm1(), pt.mo_coeff)
    occ_ref = [
        1.99999606,
        1.99997363,
        1.99745323,
        1.99184505,
        1.99031524,
        1.98913801,
        1.98016708,
        1.94721821,
        0.05637705,
        0.02061493,
        0.0158208,
        0.0110807,
    ]

    assert no.shape == mf.mo_coeff.shape
    assert abs(sum(occ) - mf.mol.nelectron) < NATOCC_TOL
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)


def test_unrestricted_natural_orbitals(tmpdir, formaldehyde_UHF):
    os.chdir(tmpdir)
    mf = formaldehyde_UHF
    rdm1mo = np.diag(mf.mo_occ[0]), np.diag(mf.mo_occ[1])
    occ, no = asf.natorbs.unrestricted_natural_orbitals(rdm1mo, mf.mo_coeff, mf.get_ovlp())
    occ_ref = [
        2.00000000,
        2.00000000,
        2.00000000,
        2.00000000,
        1.99966625,
        1.99933156,
        1.99848964,
        1.00000000,
        1.00000000,
        1.51036053e-03,
        6.68444176e-04,
        3.33750413e-04,
    ]
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)
    assert no.shape == mf.mo_coeff[1].shape
    assert abs(sum(occ) - mf.mol.nelectron) < NATOCC_TOL


def test_ao_natural_orbitals(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    mf = formaldehyde_RHF
    occ, no = asf.natorbs.ao_natural_orbitals(mf.make_rdm1(), mf.get_ovlp())
    occ_ref = [
        2.00000000,
        2.00000000,
        2.00000000,
        2.00000000,
        2.00000000,
        2.00000000,
        2.00000000,
        2.00000000,
        6.38214174e-15,
        1.81221742e-15,
        -2.43942667e-15,
        -4.23271636e-15,
    ]
    assert no.shape == mf.mo_coeff.shape
    assert abs(sum(occ) - mf.mol.nelectron) < NATOCC_TOL
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)


def test_uhf_natural_orbitals(tmpdir, formaldehyde_UHF):
    os.chdir(tmpdir)
    occ, no = asf.natorbs.uhf_natural_orbitals(formaldehyde_UHF)
    occ_ref = [
        2.00000000,
        2.00000000,
        2.00000000,
        2.00000000,
        1.99966625,
        1.99933156,
        1.99848964,
        1.00000000,
        1.00000000,
        1.51036053e-03,
        6.68444176e-04,
        3.33750413e-04,
    ]
    assert no.shape == formaldehyde_UHF.mo_coeff[1].shape
    assert abs(sum(occ) - formaldehyde_UHF.mol.nelectron) < NATOCC_TOL
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)


def test_mp2_natural_orbitals(tmpdir, formaldehyde_UHF):
    os.chdir(tmpdir)
    pt = formaldehyde_UHF.MP2().run()
    occ, no = asf.natorbs.mp2_natural_orbitals(pt)
    occ_ref = [
        1.99999692,
        1.99997373,
        1.998328,
        1.99253143,
        1.99164853,
        1.98693501,
        1.98659005,
        1.00688061,
        1.00182344,
        0.01506566,
        0.01051806,
        0.00970856,
    ]

    assert no.shape == formaldehyde_UHF.mo_coeff[1].shape
    assert abs(sum(occ) - formaldehyde_UHF.mol.nelectron) < NATOCC_TOL
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)


def test_mp2_natural_orbitals2(tmpdir, formaldehyde_MP2):
    os.chdir(tmpdir)
    mf = formaldehyde_MP2._scf  # no public getter for SCF object
    occ, no = asf.natorbs.mp2_natural_orbitals(formaldehyde_MP2)
    occ_ref = [
        1.99999606,
        1.99997363,
        1.99745323,
        1.99184505,
        1.99031524,
        1.98913801,
        1.98016708,
        1.94721821,
        0.05637705,
        0.02061493,
        0.0158208,
        0.0110807,
    ]

    assert no.shape == mf.mo_coeff.shape
    assert abs(sum(occ) - mf.mol.nelectron) < NATOCC_TOL
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)


def test_ccsd_natural_orbitals(tmpdir, formaldehyde_UHF):
    os.chdir(tmpdir)
    pt = formaldehyde_UHF.CCSD().run()
    occ, no = asf.natorbs.ccsd_natural_orbitals(pt)
    occ_ref = [
        1.99999735,
        1.99997582,
        1.99762506,
        1.98498521,
        1.98349256,
        1.9759551,
        1.96997711,
        1.01511302,
        1.00766035,
        0.02613406,
        0.02112656,
        0.0179578,
    ]

    assert no.shape == formaldehyde_UHF.mo_coeff[1].shape
    assert abs(sum(occ) - formaldehyde_UHF.mol.nelectron) < NATOCC_TOL
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)


def test_ccsd_natural_orbitals2(tmpdir, formaldehyde_RHF):
    os.chdir(tmpdir)
    pt = formaldehyde_RHF.CCSD().run()
    occ, no = asf.natorbs.ccsd_natural_orbitals(pt)
    occ_ref = [
        1.99999676,
        1.9999763,
        1.99712948,
        1.99229777,
        1.98383938,
        1.97721634,
        1.97678529,
        1.90815165,
        0.0957414,
        0.0260266,
        0.0239613,
        0.01887772,
    ]

    assert no.shape == formaldehyde_RHF.mo_coeff.shape
    assert abs(sum(occ) - formaldehyde_RHF.mol.nelectron) < NATOCC_TOL
    assert np.allclose(occ_ref, occ, atol=NATOCC_TOL)


def test_count_active_electrons():
    occ = [
        1.99999606,
        1.99997363,
        1.99745323,
        1.99184505,
        1.99031524,
        1.98913801,
        1.98016708,
        1.94721821,
        0.05637705,
        0.02061493,
        0.0158208,
        0.0110807,
    ]
    nelec = asf.natorbs.count_active_electrons(occ, list(range(len(occ))))

    assert nelec == 16


def test_select_natural_occupations_1():
    occ = [
        1.99999606,
        1.99997363,
        1.99745323,
        1.99184505,
        1.99031524,
        1.98913801,
        1.98016708,
        1.94721821,
        0.05637705,
        0.02061493,
        0.0158208,
        0.0110807,
    ]
    nelec, orbs = asf.natorbs.select_natural_occupations(np.array(occ))
    assert nelec == 2
    assert orbs == [7, 8, 9]


def test_select_natural_occupations_2():
    occ = np.array(
        [
            2.10,
            2.02,
            2.00,
            1.99,
            1.98,
            1.95,
            1.92,
            1.90,
            1.85,
            1.80,
            1.40,
            1.20,
            1.03,
            0.98,
            0.70,
            0.31,
            0.26,
            0.10,
            0.08,
            0.05,
            0.02,
            0.01,
            0.006,
            0.00001,
            -0.23,
        ]
    )

    # selection with default settings
    nel, mo_list = asf.natorbs.select_natural_occupations(occ)
    assert nel == 17
    assert np.array_equal(mo_list, np.arange(4, 21))

    # selection with maximum number of orbitals
    nel, mo_list = asf.natorbs.select_natural_occupations(occ, max_orb=9)
    assert nel == 9
    assert np.array_equal(mo_list, np.arange(8, 17))

    # selection with custom thresholds
    nel, mo_list = asf.natorbs.select_natural_occupations(occ, lower=0.25, upper=1.75, max_orb=9)
    assert nel == 5
    assert np.array_equal(mo_list, np.arange(10, 17))


def test_select_natural_occupations_3():
    occ = np.array(
        [
            0.7,
            1.2,
            1.85,
            0.98,
            0.26,
            0.00001,
            0.01,
            -0.23,
            1.4,
            2.02,
            0.08,
            0.006,
            2.1,
            1.9,
            0.02,
            1.92,
            0.1,
            0.31,
            1.95,
            1.8,
            1.98,
            0.05,
            1.99,
            2.0,
            1.03,
        ]
    )

    # selection with default settings
    nel, mo_list = asf.natorbs.select_natural_occupations(occ)
    assert nel == 17
    assert mo_list == [0, 1, 2, 3, 4, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24]

    # selection with maximum number of orbitals
    nel, mo_list = asf.natorbs.select_natural_occupations(occ, max_orb=9)
    assert nel == 9
    assert mo_list == [0, 1, 2, 3, 4, 8, 17, 19, 24]

    # selection with custom thresholds
    nel, mo_list = asf.natorbs.select_natural_occupations(occ, lower=0.25, upper=1.75, max_orb=9)
    assert nel == 5
    assert mo_list == [0, 1, 3, 4, 8, 17, 24]


def test_select_natural_occupations_4():
    occ = np.array([2.0, 1.99, 1.98, 1.97, 1.95, 0.05, 0.04, 0.03, 0.03, 0.01, 0.01, 0.0])

    nel, mo_list = asf.natorbs.select_natural_occupations(occ, lower=0.5, upper=1.5, min_orb=2)
    assert nel == 2
    assert mo_list == [4, 5]

    nel, mo_list = asf.natorbs.select_natural_occupations(occ, lower=0.02, upper=1.96, min_orb=7)
    assert nel == 6
    assert np.array_equal(mo_list, np.arange(2, 9))

    nel, mo_list = asf.natorbs.select_natural_occupations(occ, min_orb=50)
    assert nel == 10
    assert np.array_equal(mo_list, np.arange(len(occ)))


@pytest.mark.skip(reason="test will be deleted/refactored")
def test_symmetric_nat():
    mol = gto.Mole()
    mol.atom = """
    C 0.000 0.000 -0.6
    C 0.000 0.000 +0.6
    """
    mol.symmetry = True
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol)
    mf.density_fit()
    mf.kernel()

    loc_somos = lo.Boys(mol, mf.mo_coeff[0][:, [7, 8]]).kernel()
    new_coeff = mf.mo_coeff
    new_coeff[0][:, [7, 8]] = loc_somos
    new_coeff[1][:, [7, 8]] = loc_somos
    mf.mol.spin = 0
    mf.mol.build()
    # Broken symmetry solution <S^2> = 1.0135606
    mf.kernel(dm0=mf.make_rdm1(mo=new_coeff))

    pt = mf.MP2().run()
    occ, no = asf.natorbs.mp2_natural_orbitals(pt)
    natorb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, no)
    assert list(natorb_sym) == [
        "A1u",
        "A1g",
        "A1g",
        "A1u",
        "E1ux",
        "A1g",
        "E1uy",
        "E1gx",
        "E1gy",
        "A1u",
    ]


def test_all_orbital_selections():
    # Start with HF calculation for a water molecule.
    water = create_mol("water")
    mf = scf.RHF(water)
    mf.kernel()

    # Calculate single-orbital density and cumulant using CASCI.
    sf = asf.ASFCI(water, mf.mo_coeff, nel=8, norb=12)
    sf.calculate()
    orbdens = sf.one_orbital_density()
    cumulant = sf.diagonal_cumulant()

    # Iterate over multiple options to perform orbital selection.
    arglist = [
        {},
        {"pair_cumulant": cumulant},
        {"plateau_threshold": 0.0},
        {"pair_cumulant": cumulant, "cumulant_minimum_threshold": 0.01},
    ]
    for kwargs in arglist:
        # (1) get a list of viable active spaces
        result = asf.asfbase.all_orbital_selections(orbdens, **kwargs)

        # (2) Iterate over many random thresholds; check that the corresponding result is present.
        for thresh in np.random.random(10000) * np.log(4):
            nel, mo_list = asf.asfbase.entropy_selection(orbdens, threshold=thresh, **kwargs)
            norb = len(mo_list)
            if norb > 0:
                assert result[(nel, norb)] == mo_list


def test_inactive_orbital_lists():
    core, virt = asf.asfbase.inactive_orbital_lists(4, 12, np.array([4, 5, 6, 7]))
    assert np.array_equal(core, np.array([0, 1, 2, 3]))
    assert np.array_equal(virt, np.array([8, 9, 10, 11]))

    core, virt = asf.asfbase.inactive_orbital_lists(5, 15, np.array([1, 3, 6, 9]))
    assert np.array_equal(core, np.array([0, 2, 4, 5, 7]))
    assert np.array_equal(virt, np.array([8, 10, 11, 12, 13, 14]))

    core, virt = asf.asfbase.inactive_orbital_lists(0, 10, np.array([4, 5, 6, 7, 8]))
    assert np.array_equal(core, np.array([]))
    assert np.array_equal(virt, np.array([0, 1, 2, 3, 9]))
