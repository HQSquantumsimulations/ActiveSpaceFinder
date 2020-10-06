# Copyright 2020 HQS Quantum Simulations GmbH
# Reza Ghafarian Shirazi, Thilo Mast.
# reza.shirazi@quantumsimulations.de
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from numpy.linalg import eigh, multi_dot
from pyscf import symm, scf, gto, lib
import numpy as np
from functools import reduce
from typing import Tuple, Any
from asf import rdm1AO2MO
import scipy


def asserts(rdm1: np.ndarray, mo: np.ndarray, overlap: np.ndarray = None,
            mol: gto.mole.Mole = None, type: str = 'RHF', mo_a: bool = False):
    r"""Check function input.

        Args:
            rdm1 (np.ndarray):      RDM1.
            mo (np.ndarray):        Molecular orbitals.
            overlap (np.ndarray):   Method used for the calculation.
            mol (object):           mole object.
            type (str):             'RHF' or 'UHF'
            mo_a (bool):            To use alpha orbitals.

        Returns:
            rdm1 (np.ndarray):
            mo (np.ndarray):

        raises:
            Exception: UHF or RHF mistake.
            TypeError: arrays only in tuple, list or numpy array.
    """
    # mol object
    if mol is not None:
        if not isinstance(mol, gto.mole.Mole):
            raise TypeError('mol should be an instance of mole class')

    if overlap is not None:
        if not isinstance(overlap, np.ndarray):
            raise Exception('overlap should be a numpy array')

    # RDM1
    if not isinstance(rdm1, (np.ndarray, list, tuple)):
        raise TypeError('bad entery for rdm1')
    if isinstance(rdm1, (list, tuple)):
        rdm1 = np.array(rdm1)

    # MO
    if not isinstance(mo, (np.ndarray, list, tuple)):
        raise TypeError('bad entery for mo')
    if isinstance(mo, (tuple, list)):
        mo = np.array(mo)

    if type == 'UHF':
        if not rdm1.ndim == 3:
            raise Exception('Dimensions of rdm1 or mo is wrong')
    elif type == 'RHF':
        if not rdm1.ndim == 2:
            raise Exception('Dimensions of rdm1 or mo is wrong')
        if mo.ndim != 2:
            # We can select alpha orbitals?!
            if mo.ndim == 3 and mo_a:
                mo = mo[0, :, :]
            else:
                raise Exception('Bad mo dimensions, expecting 2')

    return rdm1, mo


def UNOs(mf: scf.uhf.UHF, disable_sym: bool = False) -> Tuple[np.ndarray, list]:
    r"""Calculate unrestricted natural orbitals and their occupation number.

        Args:
            mf(object):             PySCF mean-field object.
            disable_sym (bool):     Force disable symmetrizing the natural orbitals.

        Returns:
            occ (np.ndarray):       Natural orbital occupations.
            no (np.ndarray):        Natural orbitals.

        Raises:
            Exception: Only UHF.
            ValueError: wrong sum of occupation.
    """
    if not isinstance(mf, (scf.uhf.UHF)):
        raise Exception('UNOs require UHF mf object')

    S = mf.get_ovlp()
    mol = mf.mol

    # UNF UNOs

    rdm1a, rdm1b = np.diag(mf.mo_occ[0]), np.diag(mf.mo_occ[1])
    mos_a, mos_b = mf.mo_coeff

    # Transform everything into the spin-up orbital basis.
    Sab = multi_dot([mos_a.T, S, mos_b])
    rdm1 = rdm1a + multi_dot([Sab, rdm1b, Sab.T])

    no, occ = nat_orbs_rdm_rhf(rdm1, mos_a, mol=mf.mol, disable_sym=disable_sym)

    if not nat_orbs_test(occ, mol=mol):
        raise ValueError('Sum of occupation does not match number of electrons')

    return no, occ


def nat_orbs_rdm_rhf(rdm1: np.ndarray, mo: np.ndarray, mol: gto.mole.Mole = None,
                     disable_sym: bool = False, mo_a: bool = False) -> Tuple[np.ndarray, list]:
    r"""Natural orbitals from rdm1 and mos both should be RHF.

        Args:
            rdm1 (array):               1-particle reduced density matrix in MO basis.
                                        2-dimension (RHF)!
            mo (array):                 Molecular orbital coefficients.
            mol (object):               Optional in case of symmetry.
            disable_sym (bool):         Avoid symmetric diagonialization even if mol has
                                        symmetry.
            mo_a (bool):                Select alpha orbitals if UHF MOs were passed.

        Returns:
            occ (np.ndarray):           Natural orbital occupations.
            no (np.ndarray):            Natural orbitals.

        raises:
            ValueError: wrong sum of occupation.

    """
    rdm1, mo = asserts(rdm1, mo, mol=mol, mo_a=mo_a, type='RHF')

    if mol is not None:
        if (mol.symmetry is True or mol.symmetry == 1) and disable_sym is False:
            do_sym = True
        else:
            do_sym = False
    else:
        do_sym = False

    if do_sym is True:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
        eigval, eigvec = symm.eigh(rdm1, orbsym)
    else:
        eigval, eigvec = eigh(rdm1)

    occ = np.flip(eigval)
    no = np.dot(mo, np.fliplr(eigvec))

    if not nat_orbs_test(occ, mol=mol):
        raise ValueError('Sum of occupation does not match number of electrons')

    return no, occ


def nat_orbs_rdm_uhf(rdm1: np.ndarray, overlap: np.ndarray, mo: np.ndarray,
                     mol: gto.mole.Mole = None,
                     disable_sym: bool = False) -> Tuple[np.ndarray, list]:
    r"""Natural orbitals from rdm1 and mos both should be RHF.

        Args:
            rdm1 (array):               1-particle reduced density matrix in MO basis.
                                        2-dimension (RHF)!
            overlap (np.ndarray):       Overlap matrix.
            mo (array):                 Orbital coefficients.
            mol (object):               Optional in case of symmetry.
            disable_sym (bool):         Avoid symmetric diagonialization even if mol has
                                        symmetry.

        Returns:
            occ (np.ndarray):           Natural orbital occupations.
            no (np.ndarray):            Natural orbitals.

        raises:
            ValueError: wrong sum of occupation.
    """

    # Check the rdm1, mo, etc
    rdm1, mo = asserts(rdm1, mo, overlap, mol, type='UHF')

    S = overlap
    rdm1a, rdm1b = rdm1
    mos_a, mos_b = mo

    # Transform everything into the spin-up orbital basis.
    Sab = multi_dot([mos_a.T, S, mos_b])
    rdm1 = rdm1a + multi_dot([Sab, rdm1b, Sab.T])

    no, occ = nat_orbs_rdm_rhf(rdm1, mos_a, mol=mol, disable_sym=disable_sym)

    if not nat_orbs_test(occ, mol=mol):
        raise ValueError('Sum of occupation does not match number of electrons')

    return no, occ


def nat_orbs_rdm_rhf_uhf_(rdm1: np.ndarray, mo: np.ndarray,
                          mol: gto.mole.Mole = None,
                          disable_sym: bool = False) -> Tuple[np.ndarray, list]:
    r"""Avoid using this function.
        If you need to use this check your natural occupation number as you
        may get wrong number of electrons which look correct...
        Natural orbitals from rdm1 and mos both should be RHF.

        Args:
            rdm1 (array):               1-particle reduced density matrix in MO basis.
                                        2-dimension (RHF)!
            mo (array):                 Orbital coefficients.
            mol (object):               Optional in case of symmetry.
            disable_sym (bool):         Avoid symmetric diagonialization even if mol has
                                        symmetry.

        Returns:
            occ (np.ndarray):           Natural orbital occupations.
            no (np.ndarray):            Natural orbitals.

        raises:
            ValueError: wrong sum of occupation.
    """
    mos_a = mo
    # Sum rdm1 in ao basis
    rdm1a = lib.einsum('pi,ij,qj->pq', mos_a, rdm1[0, :, :], mos_a)
    rdm1b = lib.einsum('pi,ij,qj->pq', mos_a, rdm1[1, :, :], mos_a)
    rdm1 = np.add(rdm1a[:, :], rdm1b[:, :])
    # Back to MO
    inverse = np.linalg.inv(mos_a)
    rdm1 = reduce(np.dot, (inverse, rdm1, inverse.T))
    no, occ = nat_orbs_rdm_rhf(rdm1, mos_a, mol=mol, disable_sym=disable_sym)

    if not nat_orbs_test(occ, mol=mol):
        raise ValueError('Sum of occupation does not match number of electrons')

    return no, occ


def nat_orbs_rdm1AO_rhf(rdm1AO: np.ndarray, mo: np.ndarray,
                        mol: gto.mole.Mole = None, mode: int = 1) -> Tuple[np.ndarray, list]:
    r"""A wrapper for calculation of natural orbitals
        from rdm1 in AO basis.

        Args:
            rdm1AO (np.ndarray):    RDM1 in AO basis.
            mo (np.ndarray):        Molecular orbitals.
            mol (object):           mole object.
            mode (int):             Method used for the calculation.

        Returns:
            no:                     Natural orbitals.
            occ:                    Natural occupation.

        raises:
            ValueError: wrong sum of occupation.
            Exception: mode2 requires more input.
    """
    # rdm1, mo, etc check
    rdm1, mo = asserts(rdm1AO, mo, mol=mol, type='RHF')

    if mode == 1:
        rdm1MO = rdm1AO2MO(rdm1AO, mo)
        no, occ = nat_orbs_rdm_rhf(rdm1MO, mo, mol=mol)
    elif mode == 2:
        if mol is not None:
            S = mol.intor_symmetric('int1e_ovlp')
        else:
            if S is None:
                raise Exception('Either provide mol or S if you want to use mode 2')

        occ, no = scipy.linalg.eigh(rdm1AO, S, type=2)
        occ = np.flip(occ)
        no = np.fliplr(no)
    else:
        raise Exception('Wrong mode entry')

    if not nat_orbs_test(occ, mol=mol):
        raise ValueError('Sum of occupation does not match number of electrons')

    return no, occ


def nat_orbs_obj(obj: Any, disable_sym: bool = False):
    r"""Calculate natural orbitals and their occupation number.

        Args:
            obj:                    Any correlation method's PySCF object.
                                    Tested for CISD, CCSD, MP2.
            disable_sym (bool):     Avoid symmetric diagonialization even if mol has
                                    symmetry.

        Returns:
            occ (np.ndarray):       Natural orbital occupations.
            no (np.ndarray):        Natural orbitals.

        raises:
            Exception: check the errors
            ValueError: wrong sum of occupation.
    """
    S = obj.mol.intor_symmetric('int1e_ovlp')
    mo = obj.mo_coeff
    mo = np.array(mo)
    rdm1 = obj.make_rdm1(ao_repr=False)
    rdm1 = np.array(rdm1)
    mol = obj.mol

    # Do not give me CASSCF, CASCI, DMRGCI, ... if number of roots are more than 1
    # Then I have to calculate rdm for all roots etc...
    if hasattr(obj, 'fcisolver'):
        raise Exception('Do calculation for your root of choice manually')

    if isinstance(obj, (scf.uhf.UHF)):
        no, occ = UNOs(mf)
    elif isinstance(obj, (scf.rohf.ROHF, scf.rhf.RHF)):
        raise Exception('There is no natural orbitals for uncorrelated R/ROHF')
    elif rdm1.ndim == 2:
        if mo.ndim == 2:
            no, occ = nat_orbs_rdm_rhf(rdm1, mo, mol, disable_sym=disable_sym)
        else:
            raise Exception('unacceptable dimensions. rdm1: ', rdm1.ndim, 'mo: ', mo.ndim)
    elif rdm1.ndim == 3:
        if mo.ndim == 2:
            print('Warning!! UHF on RHF calculation detected!')
            no, occ = nat_orbs_rdm_rhf_uhf_(rdm1, mo, mol, disable_sym=disable_sym)
        if mo.ndim == 3:
            no, occ = nat_orbs_rdm_uhf(rdm1, S, mo, mol, disable_sym=disable_sym)
        else:
            raise Exception('unacceptable dimensions. rdm1: ', rdm1.ndim, 'mo: ', mo.ndim)
    else:
        raise Exception('unacceptable dimensions. rdm1: ', rdm1.ndim, 'mo: ', mo.ndim)

    # Assert with mol (accurate number of electrons)
    if not nat_orbs_test(occ, mol=mol):
        raise ValueError('Wrong occupation number. Electrons', mol.nelectron,
                         'sum of occupation number', sum(occ),
                         'This is usually observerd if you run RCISD on UHF')
    return no, occ


def nat_orbs_test(occ: np.ndarray, conv_tol: float = 1.0e-8, mol: gto.mole.Mole = None):
    r"""Calculate natural orbitals and their occupation number.

        Args:
            occ (np.ndarray):            List of natural orbital occupation.
            conv_tol (float):       Tolerance.
            mol (object):           mol object.

        Returns:
            bool:                   If the occupation numbers are correct or not.

        raises:
            TypeError: occupation should be an array.

    """
    if not isinstance(occ, (np.ndarray, list, tuple)):
        raise TypeError('Occupation should be an array')
    if isinstance(occ, (list, tuple)):
        occ = np.array(occ)

    if mol is None:
        scf_occ = np.round(occ)
        if abs(sum(occ) - sum(scf_occ)) < conv_tol:
            return True
        else:
            return False
    else:
        if abs(sum(occ) - mol.nelectron) < conv_tol:
            return True
        else:
            return False


def nat_orbs_singles(occ: list, tol: float = 0.11) -> list:
    r"""Find singly occupied orbitals using natural orbital occupation.

        Args:
            occ (list):       List of natural orbital occupation.
            tol (float):            Allowed deviation from one.

        Returns:
            singles (list):                idx of singly occupied orbitals.

        raises:
            TypeError: enter correct type of variable
    """
    if not isinstance(occ, (np.ndarray, list)):
        raise TypeError('occupation should be an array')

    singles = []
    for idx, i in enumerate(occ):
        if abs(i - 1) < tol:
            singles.append(idx)

    return singles


if __name__ == "__main__":
    from pyscf import mcscf, mp, ci, cc, fci

    final_spin = 0
    excited_spin = 2

    mol = gto.M(atom='C 0 0 0; C 0 0 1.24253', basis='def2-svp', spin=excited_spin, symmetry=True)
    mol.verbose = 0
    # UHF for UNOs
    mf = scf.UHF(mol).run(max_cycle=100)
    mo_new = mf.stability()[0]
    while mo_new is not mf.mo_coeff:
        mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
        mo_new = mf.stability()[0]

    no, occ = UNOs(mf)

    # UHF UMP2
    mp2 = mp.UMP2(mf).run()
    no, occ = nat_orbs_obj(mp2)

    # UHF UCISD
    mci = ci.UCISD(mf).run()
    no, occ = nat_orbs_obj(mci)

    # RHF UCCSD
    dm0 = mf.make_rdm1()
    mf = scf.RHF(mol)
    mf.kernel(dm0=dm0)
    mcc = cc.UCCSD(mf).run()
    no, occ = nat_orbs_obj(mcc)

    # Run a symmetric CASCI to test symmetric natural orbitals.
    mc = mcscf.CASCI(mf, 6, 6)
    solver1 = fci.direct_spin0_symm.FCI(mol)
    solver1.spin = final_spin
    solver1.wfnsym = 'A1g'
    solver1 = fci.addons.fix_spin(solver1, ss=final_spin / 2 * (final_spin / 2 + 1))
    solver1.nroots = 1
    solver2 = fci.direct_spin1_symm.FCI(mol)
    solver2.spin = excited_spin
    solver2.wfnsym = 'E1ux'
    solver2 = fci.addons.fix_spin(solver2, ss=excited_spin / 2 * (excited_spin / 2 + 1))
    solver2.nroots = 1
    mcscf.state_average_mix_(mc, [solver1, solver2], [0.5, 0.5])

    mc.kernel(no)
