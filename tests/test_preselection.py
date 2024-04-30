# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
import numpy as np

from asf.casci import ASFCI
from asf.dmrg import ASFDMRG
from asf.preselection import MP2NatorbPreselection, MP2PairinfoPreselection


def test_preselection_natorb_rhf(formaldehyde_RHF):
    pre = MP2NatorbPreselection(formaldehyde_RHF, min_orb=5)
    space = pre.select()

    assert space.nel == 6
    assert space.mo_list == [5, 6, 7, 8, 9, 10, 11]


def test_preselection_natorb_uhf(formaldehyde_UHF):
    pre = MP2NatorbPreselection(formaldehyde_UHF, min_orb=5)
    space = pre.select()

    assert space.nel == 6
    assert space.mo_list == [5, 6, 7, 8, 9, 10]


def test_preselection_pairinfo(formaldehyde_UHF):
    pre = MP2PairinfoPreselection(formaldehyde_UHF)
    space = pre.select()

    assert space.nel == 10
    assert space.mo_list == [i for i in range(3, 12)]


def test_preselection_init_natorb(nitrogen_RHF):
    sf = ASFCI.from_preselection(
        MP2NatorbPreselection(nitrogen_RHF, min_orb=9), nroots=2, spin_shift=0.2
    )

    assert sf.nel == 10
    assert np.array_equal(sf.mo_list, [i for i in range(2, 11)])
    assert sf.fcisolver_kwargs["nroots"] == 2
    assert sf.spin_shift == 0.2


def test_preselection_init_natorb_uhf(allyl_UHF):
    mf = allyl_UHF()
    sf = ASFDMRG.from_preselection(
        MP2NatorbPreselection(mf, lower=0.02, upper=1.98), maxM=150, tol=1e-5
    )

    assert sf.nel == 15
    assert np.array_equal(sf.mo_list, [i for i in range(4, 15)])
    assert sf.fcisolver_kwargs["maxM"] == 150
    assert sf.fcisolver_kwargs["tol"] == 1e-5


def test_preselection_init_pairinfo(allyl_UHF):
    mf = allyl_UHF()
    sf = ASFDMRG.from_preselection(MP2PairinfoPreselection(mf), maxM=150, tol=1e-5)

    assert sf.nel == 13
    assert np.array_equal(sf.mo_list, [i for i in range(5, 18)] + [22, 24])
    assert sf.fcisolver_kwargs["maxM"] == 150
    assert sf.fcisolver_kwargs["tol"] == 1e-5
