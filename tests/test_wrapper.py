from pyscf.gto import Mole
from pyscf import scf
import pytest
import numpy as np

from asf import wrapper


def test_merge_active_spaces_closedshell():
    from asf.wrapper import merge_active_spaces

    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, -0.55)), ('N', (0.0, 0.0, 0.55))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    test_spaces = [((6, [4, 5, 6, 7, 8, 9]),
                    (6, [4, 5, 6, 7, 8, 9]),
                    (6, [4, 5, 6, 7, 8, 9])),

                   ((6, [4, 5, 6, 7, 8, 9]),
                    (4, [   5, 6, 7, 8,  ]),
                    (6, [4, 5, 6, 7, 8, 9])),

                   ((2, [      6, 7,      ]),
                    (4, [4, 5,       8, 9]),
                    (6, [4, 5, 6, 7, 8, 9])),

                   (( 6, [      4, 5, 6, 7, 8, 9, 10, 11]),
                    ( 8, [2, 3,    5, 6, 7, 8           ]),
                    (10, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])),
                    
                   ((2, [6, 7]),
                    (0, [    ]),
                    (2, [6, 7])),

                   ((0, [          ]),
                    (4, [4, 5, 8, 9]),
                    (4, [4, 5, 8, 9])),
                    
                   ((0, []), (0, []), (0, []))]

    for ((nel1, mo_list1), (nel2, mo_list2), (nel_ref, mo_list_ref)) in test_spaces:
        nel, mo_list = merge_active_spaces(mol, nel1, mo_list1, nel2, mo_list2)
        assert nel == nel_ref
        assert mo_list == mo_list_ref


def test_merge_active_spaces_openshell():
    from asf.wrapper import merge_active_spaces

    mol = Mole()
    mol.atom = [('O', [0.0, 0.0, -0.49]), ('H', [0.0, 0.0, 0.49])]
    mol.basis = 'def2-SVP'
    mol.spin = 1
    mol.charge = 0
    mol.build()

    test_spaces = [((7, [1, 2, 3, 4, 5]),
                    (7, [1, 2, 3, 4, 5]),
                    (7, [1, 2, 3, 4, 5])),

                   ((5, [   2, 3, 4, 5]),
                    (7, [1, 2, 3, 4, 5]),
                    (7, [1, 2, 3, 4, 5])),

                   ((1, [   4   ]),
                    (3, [3, 4, 5]),
                    (3, [3, 4, 5])),

                   ((5, [2, 3, 4      ]),
                    (3, [   3, 4, 5, 6]),
                    (5, [2, 3, 4, 5, 6]))]

    for ((nel1, mo_list1), (nel2, mo_list2), (nel_ref, mo_list_ref)) in test_spaces:
        nel, mo_list = merge_active_spaces(mol, nel1, mo_list1, nel2, mo_list2)
        assert nel == nel_ref
        assert mo_list == mo_list_ref


def test_merge_active_spaces_exceptions():
    from asf.wrapper import merge_active_spaces

    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, -0.55)), ('N', (0.0, 0.0, 0.55))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    bad_spaces = [((5, [4, 5, 6, 7, 8, 9]),
                   (6, [4, 5, 6, 7, 8, 9])),

                  ((6, [4, 5, 6, 7, 8, 9]),
                   (4, [4, 5, 6, 7, 8, 9])),

                  ((2, [6, 7,             ]),
                   (4, [      8, 9, 10, 11])),

                  ((2, []),
                   (2, [5, 7])),

                  ((0, []),
                   (2, []))]

    for ((nel1, mo_list1), (nel2, mo_list2)) in bad_spaces:
        with pytest.raises(ValueError):
            merge_active_spaces(mol, nel1, mo_list1, nel2, mo_list2)


def test_mp2_from_scf():
    from asf.wrapper import mp2_from_scf
    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, -0.55)), ('N', (0.0, 0.0, 0.55))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    nel, mo_list, natorb  = mp2_from_scf(mf)
    assert nel == 10 
    assert np.array_equal(mo_list, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


    mf = scf.RHF(mol)
    mf.kernel()
    nel, mo_list, natorb  = mp2_from_scf(mf)

    assert nel == 10 
    assert np.array_equal(mo_list, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


def test_reorder_mos():
    from asf.wrapper import reorder_mos
    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, -0.55)), ('N', (0.0, 0.0, 0.55))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mo_list = [2, 5, 6, 7, 8, 9, 10, 11]

    a, b = reorder_mos(mol, 6, mo_list, mf.mo_coeff)
    a == [4, 5, 6, 7, 8, 9, 10, 11]
    assert np.array_equal(b[:, 4], mf.mo_coeff[:, 2])


def test_loghead():
    from asf.wrapper import loghead
    loghead('Calculating MP2 natural orbitals', 4)


def test_runasf_from_scf():
    from asf.wrapper import runasf_from_scf
    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, -0.55)), ('N', (0.0, 0.0, 0.55))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    nel, mo_idx, mo = runasf_from_scf(mf)
    assert nel == 4
    assert mo_idx == [5, 6, 7, 8]


def test_runasf_from_scf_openshell():
    from asf.wrapper import runasf_from_scf
    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, 0.58)), ('O', (0.0, 0.0, -0.58))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 1
    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()

    nel, mo_idx, _ = runasf_from_scf(mf, entropy_threshold=0.10)
    assert nel == 7
    assert mo_idx == [4, 5, 6, 7, 8, 9]

def test_do_ci():
    from asf.wrapper import do_ci, mp2_from_scf
    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, -0.55)), ('N', (0.0, 0.0, 0.55))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()
    nel, mo_list, natorb  = mp2_from_scf(mf)
    # For DMRG
    dmrgci = do_ci(mol, spin=2, molist_mp2= mo_list, natorb_mp2= natorb,
                  nel_mp2= nel, maxM=151, switch_dmrg=4)
    # check spins
    assert dmrgci.casci.nelecas[0] - dmrgci.casci.nelecas[1] == 2

    casci = do_ci(mol, spin=2, molist_mp2= mo_list, natorb_mp2= natorb,
                  nel_mp2= nel, switch_dmrg=12)

    # nelecas does not change in casci calculations even by asking for different spin
    assert casci.casci.fcisolver.spin == 2
    # Compare CASCI DMRGCI energies
    assert abs(dmrgci.casci.e_tot - casci.casci.e_tot) < 1e-7

def test_runasf_from_mole():
    from asf.wrapper import runasf_from_mole
    mol = Mole()
    mol.atom = [('N', (0.0, 0.0, -0.55)), ('N', (0.0, 0.0, 0.55))]
    mol.basis = 'def2-SVP'
    mol.charge = 0
    mol.spin = 0
    mol.build()
    nel, mo_idx, mo = runasf_from_mole(mol)
    assert nel == 4
    assert mo_idx == [5, 6, 7, 8]
