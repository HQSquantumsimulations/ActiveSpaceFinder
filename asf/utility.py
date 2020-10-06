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
import numpy as np
from functools import reduce
from pyscf import scf, gto, lib
import matplotlib.pyplot as plt
import os
from typing import Tuple, Union

################################
#    UI
################################


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


################################
#    DATA
################################

elements = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
            "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
            "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
            "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38,
            "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
            "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
            "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59,
            "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
            "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 72, "Hf": 72, "Ta": 73,
            "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
            "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87,
            "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94,
            "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101,
            "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107,
            "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113,
            "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}

################################
#    Functions
################################


def heaviest_atom(mol: gto.mole.Mole) -> int:
    r"""Takes the mol objects and returns the number of electrons on the heaviest atom.

        Args:
            mol (object):        PySCF Mole object.

        Returns:
            ele_max (int):    Number of electrons of the heaviest element in the molecule.
    """

    electrons = []
    for i in range(len(mol.elements)):
        electrons.append(elements[mol.elements[i]])

    ele_max = max(electrons)

    return ele_max


def frozen_core(mol: gto.mole.Mole) -> int:
    r"""Takes the PySCF Mole object and tells you how many core orbitals
        to freeze/keep out of the active space.

        Args:
            mol (object):	        PySCF Mole object.

        Returns:
            freeze (int):   Number of orbitals to freeze.
    """

    freeze = int(0)
    for i in range(len(mol.elements)):
        electrons = elements[mol.elements[i]]
        ecp = mol.atom_nelec_core(i)
        # Counts orbitals to freeze
        # Not a standard selection.
        if electrons < 3:
            freeze_i = 0
        elif electrons >= 3 and electrons < 11:
            freeze_i = 1
        elif electrons >= 11 and electrons < 19:
            freeze_i = 2
        elif electrons >= 19 and electrons < 37:
            freeze_i = 3
        elif electrons >= 37 and electrons < 55:
            freeze_i = 4
        elif electrons >= 55 and electrons < 87:
            freeze_i = 5
        elif electrons >= 87 and electrons < 119:
            freeze_i = 6
        # If ecp is larger ecp is the frozen part.
        if freeze < ecp:
            freeze_i = ecp
        freeze += freeze_i
    return int(freeze)


def AS_compare(mos: list) -> bool:
    r"""Compares different active spaces (list of lists) of the same size
        and checks if they are identical.
        Will sort the lists. You can input as many number of lists as you want.

        Args:
            mos (list):     The list of active spaces.
                            example: [[3, 4, 5], [3, 4, 5], [5, 4, 3]

        Returns:
            bool:           True or False.
    """
    for idx in range(len(mos)):
        mos[idx].sort()
    try:
        first = mos[0]
        return all(np.array_equal(first, rest) for rest in mos)
    except StopIteration:
        return True


def act_fao(occ: np.ndarray, ele: int) -> int:
    r"""Find number of first active orbital based on occupation numbers.
        The first one to be inside the active space.
        Counting from 0.

        Args:
            occ (list):         Occupation list.
                                If mf object is passed it will be converted to occupation.
            ele (int):          Number of electrons in active space.

        Returns:
            fao (int):          First active orbital counting from 0.
                                The orbital number given is in the active space!

        raises:
            Exception: Orbital occupations should be round.
    """
    total = 0
    if isinstance(occ, (scf.uhf.UHF, scf.rohf.ROHF, scf.rhf.RHF)):
        occ = occ.mo_occ

    # SOSCF in pyscf sometimes return the mo_occ as a tuple...
    if isinstance(occ, (tuple, list)):
        occ = np.array(occ)

    if not isinstance(ele, int):
        ele = int(np.round(ele))

    if occ.ndim == 2:
        occ = occ[0, :] + occ[1, :]
    else:
        occ = occ

    for i in range(len(occ)):
        if abs(occ[i] - np.round(occ[i])) > 1e-10:
            raise Exception('Orbital occupation should be round', i, occ[i])

    for idx, i in enumerate(reversed(occ)):
        total += i
        if abs(total - ele) < 1e-10:
            fao = len(occ) - idx + 1

    return fao - 2


def orbs_el_n(occ: np.ndarray, orbs: list, cshell: bool = False) -> int:
    r"""Find the number of electrons in the selected orbitals.

        Args:
            occ (list):             Occupation list.
                                    If you pass mf object I handle it.
                                    If mf object is passed it will be converted to occupation.
            orbs (list):            List of orbital numbers. The list should start from 0.
            cshell (bool):          Counts electrons only in closed shells.

        Returns:
            act_el (int):         Number of active electrons.
    """

    total = 0
    if isinstance(occ, (scf.uhf.UHF, scf.rohf.ROHF, scf.rhf.RHF)):
        occ = occ.mo_occ

    # SOSCF in pyscf sometimes return the mo_occ as a tuple...
    if isinstance(occ, (tuple, list)):
        occ = np.array(occ)

    if occ.ndim == 2:
        occ = occ[0, :] + occ[1, :]
    else:
        occ = occ

    # Avoid ndim conflict with list etc
    if isinstance(orbs, (list, tuple)):
        orbs = np.array(orbs)

    for j in orbs:
        if cshell is True:
            if occ[j] > 1:
                total += occ[j]
        else:
            total += occ[j]

    # Make sure it is int
    act_el = int(round(total))
    return act_el


def load_csv(name: str) -> Tuple[np.ndarray, str]:
    r"""loading a CSV file.

        Args:
            name (str):             name of the file.

        Returns:
            load (array):           numpy array of the loaded data.
            header (str):           Header of the saved data.

        raises:
            TypeError: file name should be a string.

    """
    if not isinstance(name, str):
        raise TypeError('filename should be a string.')

    # loadtxt skips header
    with open(name) as f:
        header = next(f)
        load = np.loadtxt(f, delimiter=',')

    header = header[2:]

    return load, header


def save_csv(name: str, array: np.ndarray, header: str = 'data'):
    r"""Saving a CSV file.

        Args:
            name (str):             Name of the file.
            array (array):          Array to be saved.
            header (str):           Header of the data file.

        raises:
            TypeError: pay attention to errors.

    """
    if not isinstance(array, (np.ndarray, list)):
        raise TypeError('Array should be a list or numpy array')
    if not isinstance(name, str):
        raise TypeError('filename should be a string')
    if not isinstance(header, str):
        raise TypeError('header should be a string')

    np.savetxt(name, array, delimiter=",", header=header)


def plot_selection(data: Union[list, dict], thresholds: dict, nroot: int,
                   title: str = 'single site entropy', xlabel: str = '# MO',
                   ylabel: str = '$S_{ss}$', fname: str = 'ss_entropy_', idxshift: int = 0,
                   fontsize: float = 9.0, linewidth: float = 1.1, loc: str = 'upper right',
                   fancybox: bool = True, fontsize2: float = 8.0):
    r"""Function to plot a set of point (orbital entropies, natural occupation)
        with a set of thresholds (lines) to a PNG file.

        Args:
            data (list/dic):            List of input (one dimension, x axis will be idx).
            thresholds (dict):           dictionary of thresholds.
                                        key is title of threshold, value is the float.
            nroot (int):                Number of the root being printed.
                                        This is an option which only affects the filename.
            title (str):                Title of the plot.
            xlabel (str):               Label of x axis.
            ylabel (str):               Label of y axis.
            fname (str):                Name of the file. It will be merged with root number.
            idxshift (int):             Shift idx of x axis.
            fontsize (float):           Font size in plotting.
            linewidth (float):          Line width in plotting.
            loc (str):                  Placement of plot legend.
            fontsize2 (float):          Fontsize for legend.
            fancybox (bool):            FancyBox option for legend.

        raises:
            TypeError: thresholds is a dic of lists.

    """
    plt.cla()
    if isinstance(data, (list, np.ndarray)):
        xx = np.arange(len(data)) + idxshift
        yy = [ii for ii in data]
        plt.plot(xx, yy, '*')
    elif isinstance(data, dict):
        titles = []
        yys = []
        last = 0
        for title, yy in data.items():
            xx = np.arange(len(yy)) + idxshift + last
            if xx != []:
                last = max(xx)
            yy = [ii for ii in yy]
            titles.append(title)
            yys.append(plt.plot(xx, yy, '*', label=title))
        plt.legend(loc=loc, ncol=1, fontsize=fontsize2, fancybox=fancybox)
    else:
        print('data not understood')

    for title, threshold in thresholds.items():
        if isinstance(threshold, list):
            if not len(threshold) == 2:
                raise TypeError('threshold is entered wrong')
            if not isinstance(threshold[0], (float, int)):
                raise TypeError('threshold is entered wrong')
            if not isinstance(threshold[1], str):
                raise TypeError('threshold is entered wrong')
            plt.hlines(y=threshold[0], xmin=xx[0], xmax=xx[-1], label=title,
                       color=threshold[1], linestyles='dashed', linewidth=linewidth)
            plt.text(xx[0], threshold[0], title, ha='left', va='center', fontsize=fontsize)
        elif isinstance(threshold, float):
            plt.hlines(y=threshold, xmin=xx[0], xmax=xx[-1], label=title,
                       linestyles='dashed', linewidth=linewidth)
            plt.text(xx[0], threshold, title, ha='left', va='center', fontsize=fontsize)
        else:
            print('threshold not understood')

    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname + '%s.png'
                % (nroot), dpi=300)


def visualize_mos(mol: gto.mole.Mole, mo_coeff: np.ndarray, act_mos: list, nroot: int = 1,
                  name: str = 'act_mo', keepmolden: bool = False, mo_occ: list = None,
                  mo_ene: list = None):
    r"""Function to visualize the active MOs via Jmol.
        Orbitals are counted from 0!

        Args:
            mol:		                The molecule in PySCF's gto.Mole() format.
            mo_coeff (np.ndarray):  	The MO coeff. matrix obtained via DMRG.
            act_mos (list):	            The suggested active MOs which will be plotted.
            nroot (int):		        Number of the root that is being plotted.
                                        Only for purpose of naming.
            name (str):		            Prefix of the filename for orbitals.
            keepmolden (bool):          Keep the molden file used for visualization.
            mo_occ (list):              MO_occupation to be written in molden file.
            mo_ene (list):              MO energy to be written in molden file.
    """
    # Create molden orbitals
    from pyscf.tools import molden
    molden.from_mo(mol, 'asf_mos.molden', mo_coeff, ene=mo_ene, occ=mo_occ)

    # Write jmol script
    with open('plot.spt', 'w') as plotfile:
        print('initialize;\nset background [xffffff];\nset frank off;\n',
              'set autoBond true;\nset bondRadiusMilliAngstroms 66;\n',
              'set bondTolerance 0.5;\nset forceAutoBond false;\n',
              'load asf_mos.molden;\nrotate 45 x;\nrotate 45 y;\n',
              'zoom 100;''', file=plotfile)

        for ii in act_mos:
            print('isosurface color yellow purple mo ',
                  '%s translucent; write PNG 200 "%s_%s_%s.png"'
                  % (ii + 1, name, nroot, ii), file=plotfile)
        print('exitJmol;\n', file=plotfile)
    os.system('jmol plot.spt')
    os.remove('plot.spt')


def AS_true(occ: np.ndarray, orbs: list, tol: float = 1.0e-12) -> bool:
    r"""Check if the active space makes sense.

        Args:
            occ (array):                Orbital occupation.
            orbs (list):                Orbital list.
            tol (float):                numerical 0.

        Returns:
            (bool):                     True, Good active space.
                                        False, Bad active space.
    """
    # Num elec / 2 - 2 - orbs < 0
    # At least one empty orbital
    if abs(orbs_el_n(occ, orbs) / 2 - len(orbs) - 2) < -tol:
        return False
    # Num elec - 2 > 0
    # At least one occupied orbital
    if abs(orbs_el_n(occ, orbs) - 2) < -tol:
        return False
    else:
        return True


def scf_mo_occ(mol: gto.mole.Mole) -> list:
    r"""Generates a fake scf like orbital occupation.

        Args:
            mol (object):           PySCF mol object.

        Returns:
            mo_occ (list):          Orbital occupation as list of ints.

        raises:
            ValueError:             number of spin and electrons do not match
    """
    mo_occ = []
    if not (mol.nelectron - mol.spin) % 2 == 0:
        raise ValueError("Number of electrons and spin is not sane")
    for _i in range(int(round((mol.nelectron - mol.spin) / 2))):
        mo_occ.append(2)
    for _i in range(mol.spin):
        mo_occ.append(1)
    for _i in range(mol.nao - len(mo_occ)):
        mo_occ.append(0)

    return mo_occ


def rdm1MO2AO(rdm1: np.ndarray, mo: np.ndarray) -> np.ndarray:
    r"""Convert RDM1MO to AO.

        Args:
            rdm1 (np.ndarray):          RDM1 in MO basis.
            mo (np.ndarray):            MO coefficients.

        Returns:
            rdm1 (np.ndarray):          RDM1 in AO basis.
    """

    rdm1AO = lib.einsum('pi,ij,qj->pq', mo, rdm1, mo.conj())
    return rdm1AO


def rdm2MO2AO(rdm2: np.ndarray, mo: np.ndarray):
    r"""Convert RDM2MO to AO.

        Args:
            rdm2 (np.ndarray):          RDM1 in MO basis.
            mo (np.ndarray):            MO coefficients.

        Returns:
            rdm2 (np.ndarray):          RDM1 in AO basis.
    """
    rdm2 = rdm2.transpose(1, 0, 3, 2)
    mo_Con = mo.conj()
    rdm2AO = lib.einsum('ijkl,pi,qj,rk,sl->pqrs', rdm2, mo, mo_Con, mo, mo_Con)
    return rdm2AO


def rdm1AO2MO(rdm1: np.ndarray, mo: np.ndarray) -> np.ndarray:
    r"""Convert RDM1AO to MO.

        Args:
            rdm1 (np.ndarray):          RDM1 in AO basis.
            mo (np.ndarray):            MO coefficients.

        Returns:
            rdm1 (np.ndarray):          RDM1 in MO basis.
    """
    inverse = np.linalg.inv(mo)
    rdm1MO = reduce(np.dot, (inverse, rdm1, inverse.T))

    return rdm1MO


def del_DMRG_temp_(mc):
    import shutil

    print("Deleting the following files:/n",
          mc.fcisolver.scratchDirectory + '/node0/n',
          mc.fcisolver.scratchDirectory + '/' + mc.fcisolver.runtimeDir,)
    shutil.rmtree(mc.fcisolver.scratchDirectory + '/node0')
    shutil.rmtree(mc.fcisolver.scratchDirectory + '/' + mc.fcisolver.runtimeDir)


def ovp_check(mol1: gto.mole.Mole, mol2: gto.mole.Mole, mo1: np.ndarray,
              mo2: np.ndarray, tol: float = 0.35, mo_range: list = None) -> list:
    r"""Compare two sets of orbitals then return the list of orbital indices
        where orbitals of the same index match whith each other.

        Args:
            mol1 (object):      mol of system 1.
            mol2 (object):      mol of system 2.
            mo1 (np.ndarray):   molecular orbitals of system 1.
            mo2 (np.ndarray):   molecular orbitals of system 2.
            tol (float):        tolerance.
            mo_range (list):    range of orbitals to check.

        Returns:
            match (list):              List of matched orbital indices.

        raises:
            TypeError: MO both should be of the same size.
    """
    from pyscf.tools import mo_mapping
    if not mo1.shape[0] == mo2.shape[0]:
        raise TypeError('Both MOs should be of the same dimensions')
    if mo_range is None:
        mo_range = list(range(mo1.shape[0]))

    match = []
    idx = np.array(mo_mapping.mo_map(mol1, mo1, mol2, mo2, tol=tol)[0])
    for i in mo_range:
        idxi = idx[idx[:, 0] == i]
        for j in idxi:
            if set(j) == set([i, i]):
                match.append(i)
    return match


if __name__ == "__main__":
    from pyscf import gto, mp

    mos = [[5, 4, 6], [4, 5, 6], [6, 5, 4]]
    assert AS_compare(mos)

    occ = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
    assert act_fao(occ, 4) == 5

    occ = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    assert act_fao(occ, 4) == 5

    assert orbs_el_n(occ, [4, 5, 6]) == 6

    mol = gto.Mole()
    mol.atom = """
    C          0.00000        0.00000        0.00000
    C          0.00000        0.00000        1.20000
    """
    mol.basis = 'def2-tzvp'
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 2
    mol.max_memory = 2500
    mol.symmetry = False
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    assert act_fao(mf, 4) == 4
    assert orbs_el_n(mf, [4, 5, 6]) == 4
    # CISD natural orbitals
    mp2 = mp.MP2(mf).run()

    from nat_orbs import nat_orbs_obj
    no, occ = nat_orbs_obj(mp2)

    # Check rdm conversions
    rdm1AO = mp2.make_rdm1(ao_repr=True)
    rdm1MO = rdm1AO2MO(rdm1AO, mf.mo_coeff)
    assert np.trace(rdm1MO) - mol.nelectron < 1.0e-12

    rdm1AO_2 = rdm1MO2AO(rdm1MO, mf.mo_coeff)
    assert np.allclose(rdm1AO, rdm1AO_2)

    rdm2AO_1 = mp2.make_rdm2(ao_repr=True)
    rdm2MO = mp2.make_rdm2(ao_repr=False)
    rdm2AO_2 = rdm2MO2AO(rdm2MO, mf.mo_coeff)
    assert np.allclose(rdm2AO_2, rdm2AO_1)

    from asf import asf

    ASF = asf()
    sel_mos = ASF.select_mos_occ(occ)
