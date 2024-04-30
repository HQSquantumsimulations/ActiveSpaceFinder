# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.
from typing import cast
from asf.pairinfo import ActiveSpacesDict

import pytest
from asf.casci import ASFCI
from asf.preselection import MP2NatorbPreselection
from pyscf.mp import MP2
from pyscf.mp.dfump2_native import DFUMP2
from pyscf.mp.mp2 import MP2 as MP2class
from pyscf.mp.ump2 import UMP2 as UMP2Class
from pyscf.scf import RHF, UHF
from pyscf.scf.hf import RHF as RHFclass
from pyscf.scf.uhf import UHF as UHFclass

from .fixtures.molecules import create_mol


@pytest.fixture(scope="package")
def formaldehyde_RHF() -> RHFclass:
    formaldehyde = create_mol("formaldehyde")
    mf = RHF(formaldehyde)
    mf.density_fit()
    mf.run(conv_tol=1e-11)
    return mf


@pytest.fixture(scope="package")
def formaldehyde_UHF() -> UHFclass:
    formaldehyde_triplet = create_mol("formaldehyde", spin=2)
    mf = UHF(formaldehyde_triplet)
    mf.density_fit()
    mf.run(conv_tol=1e-11)
    return mf


@pytest.fixture(scope="package")
def formaldehyde_MP2(formaldehyde_RHF: RHFclass) -> MP2class:
    pt = MP2(formaldehyde_RHF).run()
    return pt


@pytest.fixture(scope="package")
def ammonia_RHF() -> RHFclass:
    ammonia = create_mol("ammonia")
    return RHF(ammonia).run(conv_tol=1e-12)


@pytest.fixture(scope="package")
def OH_radical_UHF() -> UHFclass:
    OH_radical = create_mol("OH_radical")
    mf = UHF(OH_radical).run(conv_tol=1e-12)
    return mf


@pytest.fixture(scope="package")
def OH_radical_DFUMP2(OH_radical_UHF: UHFclass) -> DFUMP2:
    # Use a deliberately large auxiliary set to reduce deviations between MP2 and RI-MP2.
    pt = DFUMP2(OH_radical_UHF, auxbasis="def2-QZVPPD-RI").run()
    return pt


@pytest.fixture(scope="package")
def OH_radical_UMP2(OH_radical_UHF: UHFclass) -> UMP2Class:
    pt = MP2(OH_radical_UHF).run()
    return cast(UMP2Class, pt)


@pytest.fixture(scope="package")
def nitrogen_RHF() -> RHFclass:
    nitrogen = create_mol("nitrogen")
    mf = RHF(nitrogen).run()
    return mf


@pytest.fixture(scope="package")
def allyl_UHF() -> UHFclass:
    """
    Perform UHF calculation for the allyl radical.
    """
    mol = create_mol("allyl")
    mf = UHF(mol).run(conv_tol=1e-12)
    mf.stability()
    return mf


@pytest.fixture(scope="package")
def ethene_RHF() -> RHFclass:
    ethene = create_mol("ethene")
    mf = RHF(ethene).run()
    return mf


@pytest.fixture(scope="package")
def ethene_ASFCI(ethene_RHF) -> ASFCI:
    pre = MP2NatorbPreselection(ethene_RHF, min_orb=4, max_orb=11, lower=0.01, upper=1.99)
    space = pre.select()
    sf = ASFCI(ethene_RHF.mol, space.mo_coeff, nel=space.nel, mo_list=space.mo_list)
    sf.calculate()
    return sf


@pytest.fixture(scope="package")
def ethene_unfiltered_spaces(ethene_ASFCI) -> ActiveSpacesDict:
    return ethene_ASFCI.unfiltered_pairinfo_spaces()


@pytest.fixture(scope="package")
def ethene_filtered_spaces(ethene_ASFCI) -> ActiveSpacesDict:
    return ethene_ASFCI.find_many()


@pytest.fixture(scope="package")
def OH_radical_ASFCI(OH_radical_UHF) -> ASFCI:
    pre = MP2NatorbPreselection(OH_radical_UHF)
    sf = ASFCI.from_preselection(pre)
    sf.calculate()
    return sf


@pytest.fixture(scope="package")
def OH_radical_unfiltered_spaces(OH_radical_ASFCI) -> ActiveSpacesDict:
    return OH_radical_ASFCI.unfiltered_pairinfo_spaces()
