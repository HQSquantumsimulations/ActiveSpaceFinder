import os


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


def test_setMOList():
    import asf
    from pyscf import scf
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.kernel()
    mo_list = [6, 7, 8]
    asfci = asf.ASFCI(mol, mf.mo_coeff, nel=4, mo_list=mo_list)
    assert asfci.norb == 3


def test_runCI():
    import asf
    from pyscf import scf
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    asfci = asf.ASFCI(mol, mf.mo_coeff, nel=4, mo_list=[6, 7, 8])
    asfci.runCI(conv_tol=1e-11)
    assert abs(asfci.casci.e_tot + 113.69506678408428) < 1e-9
    assert asfci.casci.fcisolver.conv_tol == 1e-11


def test_getOrbDens():
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    asfci = asf.ASFCI(mol, mf.mo_coeff, nel=4, mo_list=[6, 7, 8, 9])
    asfci.runCI(conv_tol=1e-11)
    orbd = np.array(asfci.getOrbDens())

    orbd_o = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                      [5.66923410e-02, 1.34973881e-04, 1.34973881e-04, 9.43037711e-01],
                      [1.69640949e-03, 0.00000000e+00, 0.00000000e+00, 9.98303591e-01],
                      [9.42286343e-01, 1.34973881e-04, 1.34973881e-04, 5.74437093e-02],
                      [9.99054959e-01, 0.00000000e+00, 0.00000000e+00, 9.45041172e-04],
                      [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                      [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    for x, i in enumerate(orbd_o):
        assert(np.allclose(orbd[x], i, atol=asfci.casci.fcisolver.conv_tol))
    assert orbd.shape == (12, 4)


def test_calcEntropy():
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    asfci = asf.ASFCI(mol, mf.mo_coeff, nel=4, mo_list=[6, 7, 8, 9])
    asfci.runCI(conv_tol=1e-11)
    S1 = asfci.calcEntropy()
    print(S1.shape)
    S1_o = [0., 0., 0., 0., 0., 0., 0.22042718, 0.01251678,
            0.22253437, 0.00752613, 0., 0.]
    assert(np.allclose(S1, S1_o, atol=asfci.casci.fcisolver.conv_tol))


def test_runDMRG(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    asfdmrg = asf.ASFDMRG(mol, mf.mo_coeff, 4, 4)
    asfdmrg.runDMRG(maxM=150, tol=1e-10)
    assert (abs(asfdmrg.casci.e_tot + 113.69630979261541) < 1e-7)
    assert asfdmrg.casci.fcisolver.tol == 1e-10


def test_getOrbDens_dmrg(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    asfdmrg = asf.ASFDMRG(mol, mf.mo_coeff, 4, 4)
    asfdmrg.runDMRG(maxM=150, tol=1e-10)
    orbd = asfdmrg.getOrbDens()
    orbd_o = [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
              [5.66923951e-02, 1.34971806e-04, 1.34971806e-04, 9.43037661e-01],
              [1.69640193e-03, 1.21014310e-14, 1.09912079e-14, 9.98303598e-01],
              [9.42286300e-01, 1.34971806e-04, 1.34971806e-04, 5.74437561e-02],
              [9.99054959e-01, 1.12169388e-14, 1.12165052e-14, 9.45040944e-04],
              [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
              [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]
    for x, i in enumerate(orbd_o):
        assert(np.allclose(orbd[x], i, atol=1e-7))
    assert orbd.shape == (12, 4)


def test_selectOrbitals(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    sf = asf.ASFCI(mol, mf.mo_coeff)
    sf.runCI()
    nelec, orbs = sf.selectOrbitals()
    print(orbs)
    assert nelec == 2
    assert list(orbs) == [6, 8]


def test_selectOrbitals_roots_CI(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    sf = asf.ASFCI(mol, mf.mo_coeff)
    sf.runCI(nroots=2)
    nelec, orbs = sf.selectOrbitals(root=0)
    assert nelec == 2
    assert list(orbs) == [6, 8]
    # entropy of this root
    # entropy_ref = [5.80858656e-05, 2.75277339e-04, 5.36580738e-02, 8.86412207e-02,
    #                8.20611595e-02, 9.89239187e-02, 2.67710551e-01, 1.01588892e-01,
    #                2.75214653e-01, 1.13480251e-01, 1.31990272e-01, 1.11464837e-01]
    # assert np.allclose(sf.entropy, entropy_ref, atol=1e-7)
    nelec, orbs = sf.selectOrbitals(root=1)
    assert nelec == 6
    assert list(orbs) == [4, 6, 7, 8, 9, 10]
    # entropy_ref = [4.29475301e-05, 2.84879828e-04, 5.31488611e-02, 1.14860646e-01,
    #                3.99563236e-01, 9.55310707e-02, 3.16982530e-01, 9.57102030e-01,
    #                9.06211823e-01, 1.40129203e-01, 1.36965856e-01, 1.04377069e-01]
    # assert np.allclose(sf.entropy, entropy_ref, atol=1e-7)


def test_selectOrbitals_roots_DMRG(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    sf = asf.ASFDMRG(mol, mf.mo_coeff)
    sf.runDMRG(nroots=2)
    nelec, orbs = sf.selectOrbitals(root=0)
    assert nelec == 2
    assert list(orbs) == [6, 8]
    nelec, orbs = sf.selectOrbitals(root=1)
    assert nelec == 6
    assert list(orbs) == [4, 6, 7, 8, 9, 10]


def test_restrictedNaturalOrbitals(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.kernel()
    pt = mf.MP2().run()
    occ, no = asf.natorbs.restrictedNaturalOrbitals(pt.make_rdm1(), pt.mo_coeff)
    occ_ref = [1.99999606, 1.99997363, 1.99745323, 1.99184505, 1.99031524, 1.98913801,
               1.98016708, 1.94721821, 0.05637705, 0.02061493, 0.0158208, 0.0110807]

    assert no.shape == mf.mo_coeff.shape
    assert abs(sum(occ) - mol.nelectron) < 1e-7
    assert np.allclose(occ_ref, occ, atol=1e-7)


def test_unrestrictedNaturalOrbitals(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol(spin=2)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-11
    mf.density_fit()
    mf.kernel()
    rdm1mo = np.diag(mf.mo_occ[0]), np.diag(mf.mo_occ[1])
    occ, no = asf.natorbs.unrestrictedNaturalOrbitals(rdm1mo, mf.mo_coeff, mf.get_ovlp())
    occ_ref = [2.00000000, 2.00000000, 2.00000000, 2.00000000,
               1.99966625, 1.99933156, 1.99848964, 1.00000000,
               1.00000000, 1.51036053e-03, 6.68444176e-04, 3.33750413e-04]
    print(max(occ_ref - occ))
    assert np.allclose(occ_ref, occ, atol=1e-7)
    assert(no.shape == mf.mo_coeff[1].shape)
    assert abs(sum(occ) - mol.nelectron) < 1e-7


def test_AOnaturalOrbitals(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    occ, no = asf.natorbs.AOnaturalOrbitals(mf.make_rdm1(), mf.get_ovlp())
    occ_ref = [ 2.00000000,  2.00000000,  2.00000000,  2.00000000,
                2.00000000,  2.00000000,  2.00000000,  2.00000000,
                6.38214174e-15,  1.81221742e-15, -2.43942667e-15, -4.23271636e-15]
    assert no.shape == mf.mo_coeff.shape
    assert abs(sum(occ) - mol.nelectron) < 1e-7
    assert np.allclose(occ_ref, occ, atol=1e-7)


def test_UHFNaturalOrbitals(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol(spin=2)
    mf = scf.UHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    occ, no = asf.natorbs.UHFNaturalOrbitals(mf)
    occ_ref = [2.00000000, 2.00000000, 2.00000000, 2.00000000,
               1.99966625, 1.99933156, 1.99848964, 1.00000000,
               1.00000000, 1.51036053e-03, 6.68444176e-04, 3.33750413e-04]
    assert no.shape == mf.mo_coeff[1].shape
    assert abs(sum(occ) - mol.nelectron) < 1e-7
    assert np.allclose(occ_ref, occ, atol=1e-7)


def test_MP2NaturalOrbitals(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol(spin=2)
    mf = scf.UHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    pt = mf.MP2().run()
    occ, no = asf.natorbs.MP2NaturalOrbitals(pt)
    occ_ref = [1.99999692, 1.99997373, 1.998328, 1.99253143, 1.99164853, 1.98693501,
               1.98659005, 1.00688061, 1.00182344, 0.01506566, 0.01051806, 0.00970856]

    assert no.shape == mf.mo_coeff[1].shape
    assert abs(sum(occ) - mol.nelectron) < 1e-7
    assert np.allclose(occ_ref, occ, atol=1e-7)


def test_MP2NaturalOrbitals2(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    pt = mf.MP2().run()
    occ, no = asf.natorbs.MP2NaturalOrbitals(pt)
    occ_ref = [1.99999606, 1.99997363, 1.99745323, 1.99184505, 1.99031524, 1.98913801,
               1.98016708, 1.94721821, 0.05637705, 0.02061493, 0.0158208, 0.0110807 ]

    assert no.shape == mf.mo_coeff.shape
    assert abs(sum(occ) - mol.nelectron) < 1e-7
    assert np.allclose(occ_ref, occ, atol=1e-7)


def test_CCSDNaturalOrbitals(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol(spin=2)
    mf = scf.UHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    pt = mf.CCSD().run()
    occ, no = asf.natorbs.CCSDNaturalOrbitals(pt)
    occ_ref = [1.99999735, 1.99997582, 1.99762506, 1.98498521, 1.98349256, 1.9759551,
               1.96997711, 1.01511302, 1.00766035, 0.02613406, 0.02112656, 0.0179578 ]

    assert no.shape == mf.mo_coeff[1].shape
    assert abs(sum(occ) - mol.nelectron) < 1e-7
    assert np.allclose(occ_ref, occ, atol=1e-7)


def test_CCSDNaturalOrbitals2(tmpdir):
    os.chdir(tmpdir)
    import asf
    from pyscf import scf
    import numpy as np
    mol = get_mol()
    mf = scf.RHF(mol)
    mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    pt = mf.CCSD().run()
    occ, no = asf.natorbs.CCSDNaturalOrbitals(pt)
    print(occ)
    occ_ref = [1.99999676, 1.9999763,  1.99712948, 1.99229777, 1.98383938, 1.97721634,
               1.97678529, 1.90815165, 0.0957414,  0.0260266,  0.0239613,  0.01887772]

    assert no.shape == mf.mo_coeff.shape
    assert abs(sum(occ) - mol.nelectron) < 1e-7
    assert np.allclose(occ_ref, occ, atol=1e-7)


def test_countActiveElectrons():
    import asf
    occ = [1.99999606, 1.99997363, 1.99745323, 1.99184505, 1.99031524, 1.98913801,
               1.98016708, 1.94721821, 0.05637705, 0.02061493, 0.0158208, 0.0110807]
    nelec = asf.natorbs.countActiveElectrons(occ, list(range(len(occ))))
    
    assert nelec==16
    mol = get_mol()


def test_selectNaturalOccupations_1():
    import asf
    import numpy as np
    occ = [1.99999606, 1.99997363, 1.99745323, 1.99184505, 1.99031524, 1.98913801,
               1.98016708, 1.94721821, 0.05637705, 0.02061493, 0.0158208, 0.0110807]
    nelec, orbs = asf.natorbs.selectNaturalOccupations(np.array(occ))
    assert nelec == 2
    assert list(orbs) == [7, 8, 9]


def test_selectNaturalOccupations_2():
    from asf.natorbs import selectNaturalOccupations
    import numpy as np
    occ = np.array([2.10, 2.02, 2.00, 1.99, 1.98, 1.95, 1.92, 1.90, 1.85, 1.80, 1.40, 1.20, 1.03,
                    0.98, 0.70, 0.31, 0.26, 0.10, 0.08, 0.05, 0.02, 0.01, 0.006, 0.00001, -0.23])

    # selection with default settings
    nel, mo_list = selectNaturalOccupations(occ)
    assert nel == 17
    assert np.array_equal(mo_list, np.arange(4, 21))

    # selection with maximum number of orbitals
    nel, mo_list = selectNaturalOccupations(occ, max_orb=9)
    assert nel == 9
    assert np.array_equal(mo_list, np.arange(8, 17))

    # selection with custom thresholds
    nel, mo_list = selectNaturalOccupations(occ, lower=0.25, upper=1.75, max_orb=9)
    assert nel == 5
    assert np.array_equal(mo_list, np.arange(10, 17))


def test_selectNaturalOccupations_3():
    from asf.natorbs import selectNaturalOccupations
    import numpy as np
    occ = np.array([0.7, 1.2, 1.85, 0.98, 0.26, 0.00001, 0.01, -0.23, 1.4, 2.02, 0.08, 0.006, 2.1,
                    1.9, 0.02, 1.92, 0.1, 0.31, 1.95, 1.8, 1.98, 0.05, 1.99, 2.0, 1.03])

    # selection with default settings
    nel, mo_list = selectNaturalOccupations(occ)
    assert nel == 17
    assert np.array_equal(mo_list, [0, 1, 2, 3, 4, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24])

    # selection with maximum number of orbitals
    nel, mo_list = selectNaturalOccupations(occ, max_orb=9)
    assert nel == 9
    assert np.array_equal(mo_list, [0, 1, 2, 3, 4, 8, 17, 19, 24])

    # selection with custom thresholds
    nel, mo_list = selectNaturalOccupations(occ, lower=0.25, upper=1.75, max_orb=9)
    assert nel == 5
    assert np.array_equal(mo_list, [0, 1, 3, 4, 8, 17, 24])


def test_selectNaturalOccupations_4():
    from asf.natorbs import selectNaturalOccupations
    import numpy as np
    occ = np.array([2.0, 1.99, 1.98, 1.97, 1.95, 0.05, 0.04, 0.03, 0.03, 0.01, 0.01, 0.0])

    nel, mo_list = selectNaturalOccupations(occ, lower=0.5, upper=1.5, min_orb=2)
    assert nel == 2
    assert np.array_equal(mo_list, [4, 5])

    nel, mo_list = selectNaturalOccupations(occ, lower=0.02, upper=1.96, min_orb=7)
    assert nel == 6
    assert np.array_equal(mo_list, np.arange(2, 9))

    nel, mo_list = selectNaturalOccupations(occ, min_orb = 50)
    assert nel == 10
    assert np.array_equal(mo_list, np.arange(len(occ)))

def test_symmetric_nat():
    from pyscf import gto, scf, symm
    import asf
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
    from pyscf import lo
    loc_somos = lo.Boys(mol, mf.mo_coeff[0][:, [7, 8]]).kernel()
    new_coeff = mf.mo_coeff
    new_coeff[0][:, [7, 8]] = loc_somos
    new_coeff[1][:, [7, 8]] = loc_somos
    mf.mol.spin = 0
    mf.mol.build()
    # Broken symmetry solution <S^2> = 1.0135606
    mf.kernel(dm0=mf.make_rdm1(mo=new_coeff))

    pt = mf.MP2().run()
    occ, no = asf.natorbs.MP2NaturalOrbitals(pt)
    natorb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, no)
    print(natorb_sym)
    assert list(natorb_sym) == ['A1u', 'A1g', 'A1g', 'A1u', 'E1ux', 'A1g', 'E1uy', 'E1gx', 'E1gy', 'A1u']


def test_inactive_orbital_lists():
    import asf
    import numpy as np

    core, virt = asf.asf.inactive_orbital_lists(4, 12, np.array([4, 5, 6, 7]))
    assert np.array_equal(core, np.array([0, 1, 2, 3]))
    assert np.array_equal(virt, np.array([8, 9, 10, 11]))

    core, virt = asf.asf.inactive_orbital_lists(5, 15, np.array([1, 3, 6, 9]))
    assert np.array_equal(core, np.array([0, 2, 4, 5, 7]))
    assert np.array_equal(virt, np.array([8, 10, 11, 12, 13, 14]))

    core, virt = asf.asf.inactive_orbital_lists(0, 10, np.array([4, 5, 6, 7, 8]))
    assert np.array_equal(core, np.array([]))
    assert np.array_equal(virt, np.array([0, 1, 2, 3, 9]))
