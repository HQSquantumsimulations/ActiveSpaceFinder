# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Utility functions"""

import tempfile
import os
from shutil import which
import subprocess  # noqa: S404
from contextlib import contextmanager
from typing import List, Tuple, Union, Generator, Sequence

import numpy as np

from pyscf.gto import Mole
from pyscf.tools import molden
from pyscf.lib import parameters
from pyscf.mcscf.casci import CASCI as CASCIclass


@contextmanager
def tempname(dir: str = parameters.TMPDIR, suffix: str = None) -> Generator[str, None, None]:
    """
    Context manager that creates a temporary file name, and deletes the file at exit.

    In contrast to tempfile.TemporaryFile, it only returns the name and not a file object.
    This permits the user to hand it over to functions which expect a file name instead of a file
    object, hand it over to other programs for opening and closing, etc.

    Args:
        dir:    Directory to store the temporary file.
        suffix: Suffix for the temporary file name.

    Yields:
        The name of the temporary file.
    """
    # First we create the temporary file on disk.
    _, filename = tempfile.mkstemp(dir=dir, suffix=suffix)
    try:
        # Provides the file name in "with tempname(...) as ...:".
        yield filename
    finally:
        # Make sure to clean up in the end.
        os.remove(filename)


def pictures_Jmol(mol: Mole,
                  mo_coeff: np.ndarray,
                  mo_list: Union[List[int], np.ndarray],
                  name: str = "mo",
                  cutoff: float = 0.04,
                  ene: Union[np.ndarray, List[float]] = None,
                  occ: Union[np.ndarray, List[float]] = None,
                  rotate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  script_options: str = '',
                  jmol_exec: str = 'jmol',
                  jmol_opts: str = '--nodisplay',
                  xvfb_cmd: str = None) -> None:
    """
    Create pictures of orbitals using Jmol.

    Note that the indices of orbitals in mo_list start from zero, whereas Jmol counts orbitals
    starting from one.

    Args:
        mol:            The molecule as an instance of pyscf.gto.Mole.
        mo_coeff:       Matrix of MO coefficients.
        mo_list:        Indices of MOs to be plotted (indexing based on zero).
        name:           File names of pictures will be: [name]_[MO index].png
        cutoff:         Isosurface cutoff for orbital rendering.
        ene:            If provided, orbital energies will included in the MO titles.
        occ:            If provided, orbital occupancies will be included in the MO titles.
        rotate:         Rotation angle about the (x-axis, y-axis, z-axis) in degrees.
        script_options: A string with arbitrary input to be included in the Jmol script.
        jmol_exec:      Name of the Jmol executable.
        jmol_opts:      String containing to be provided to Jmol.
        xvfb_cmd:       X virtual framebuffer command.
                        None (default): Use 'xvfb-run' if it is available.
                        string:         Enforces the supplied name for the xfvb-run command.
                        '':             Call Jmol without xvfb-run.
    """
    # Set xvfb option.
    if xvfb_cmd is None:
        xvfb_cmd = 'xvfb-run' if which('xvfb-run') else ''

    # Apparently, Jmol does not work properly if the script does not have an .spt ending.
    with tempname(suffix='.molden') as molden_filename, tempname(suffix='.spt') as jmol_filename:

        # Dump the orbitals into the temporary molden file.
        molden.from_mo(mol, molden_filename, mo_coeff, ene=ene, occ=occ)

        mo_titleoptions = ['MO %I of %N']
        if mol.symmetry:
            mo_titleoptions.append('Symmetry: %S')
        if ene is not None:
            mo_titleoptions.append('Energy: %.4E %U')
        if occ is not None:
            mo_titleoptions.append('Occupancy: %O')
        mo_titleformat = '|'.join(mo_titleoptions)

        # Open Jmol script file (temporary) for writing.
        f = open(jmol_filename, 'w')
        # Start off by loading the molden file (otherwise some settings are not applied).
        f.write('load "{0:s}" \n'.format(molden_filename))
        # White background, no "Jmol" label in the picture.
        f.write('background white\n')
        f.write('frank off\n')
        # Represent molecule as sticks with radius 0.1 Angstrom.
        f.write('spacefill off\n')
        f.write('wireframe 0.1\n')
        # Settings to render MOs.
        f.write('mo fill nomesh translucent\n')
        f.write('mo color yellow purple\n')
        f.write('mo resolution 20\n')
        f.write('mo cutoff {0:f}\n'.format(cutoff))
        # labelling in the picture
        f.write('mo titleformat "{0:s}"\n'.format(mo_titleformat))
        # Rotate the molecule into a different orientation.
        f.write('rotate {0:f} x\n'.format(rotate[0]))
        f.write('rotate {0:f} y\n'.format(rotate[1]))
        f.write('rotate {0:f} z\n'.format(rotate[2]))
        # Give the user a chance to override any settings.
        f.write(script_options + '\n')
        # Now iterate over the MOs and dump PNG files.
        for i in mo_list:
            f.write('mo {0:d}\n'.format(i + 1))
            f.write('write image png "{0:s}_{1:d}.png"\n'.format(name, i))
        f.close()

        # Execute Jmol using the files we have created previously.
        if xvfb_cmd:
            runargs = [xvfb_cmd, jmol_exec, jmol_opts, jmol_filename]
        else:
            runargs = [jmol_exec, jmol_opts, jmol_filename]
        subprocess.run(runargs, shell=False)  # noqa: S603


def corresponding_orbitals(mol: Mole,
                           mo_coeff1: np.ndarray,
                           mo_coeff2: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform corresponding orbital transformation for two orbital sets.

    The original idea dates back to Amos and Hall (https://doi.org/10.1098/rspa.1961.0175).
    See also the work of Neese (https://doi.org/10.1016/j.jpcs.2003.11.015).

    Args:
        mol:        Mole object with the basis set and molecular data.
        mo_coeff1:  First set of MO coefficients.
        mo_coeff2:  Second set of MO coefficients.

    Returns:
        Tuple (singular values, first corresponding orbital set, second corresponding orbital set)
    """
    # Overlap of atomic orbitals.
    ovlp_ao = mol.intor_symmetric('int1e_ovlp')

    # Overlap of the two MO sets.
    ovlp_mo = np.linalg.multi_dot([mo_coeff1.T, ovlp_ao, mo_coeff2])

    # SVD of the MO overlap.
    U, sigma, Vt = np.linalg.svd(ovlp_mo)

    # The corresponding orbitals.
    co_coeff1 = mo_coeff1.dot(U)
    co_coeff2 = mo_coeff2.dot(Vt.T)

    return sigma, co_coeff1, co_coeff2


def compare_active_spaces(cas: CASCIclass,
                          mo_guess: np.ndarray,
                          mo_list: Union[Sequence[int], np.ndarray],
                          full_result: bool = False) -> \
        Union[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Given a CASSCF (or CASCI) object, calculate its consistency with some guess orbitals.

    This function employs the corresponding orbital transformation.

    One way to use the function is just to examine the smallest singular value. If it is close to
    1.0, the orbitals are likely consistent. With a value of 0.0 they are most probably
    inconsistent.

    Another way to use the function is to obtain the full set of corresponding orbitals.

    Args:
        cas:            Instance of CASSCF (or CASCI).
        mo_guess:       Guess orbitals that were used for CASSCF.
        mo_list:        List of guess orbitals that were included in the active space.
        full_result:    Return the full set of corresponding orbitals if set to True.
                        Otherwise, just provide the smallest singular value.

    Returns:
        The smallest singular value if full_result is False.
        Tuple (singular values, corresponding guess MOs, corresponding CASSCF MOs) otherwise.
    """
    mol = cas.mol

    # Truncated active space MO coefficients of the guess.
    guess_active = mo_guess[:, mo_list]

    # Truncated active space MO coefficients of the CAS object.
    ncore = cas.ncore
    ncas = cas.ncas
    final_active = cas.mo_coeff[:, ncore:ncore + ncas]

    # Perform corresponding orbital transformation of the active spaces.
    sigma, co_guess, co_cas = corresponding_orbitals(mol, guess_active, final_active)

    if full_result:
        return (sigma, co_guess, co_cas)
    else:
        return float(min(sigma))


def rdm1s_from_rdm12(N: int,
                     S: float,
                     rdm1: np.ndarray,
                     rdm2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the alpha and beta components of the 1-RDM from the spin-free 1-RDM and 2-RDM.

    Implements equation 3 from Gidofalvi, Shepard, IJQC 109 (2009), 3552
    DOI: 10.1002/qua.22320

    Args:
        N:      Number of electrons.
        S:      The spin (0.0, 0.5, 1.0, ...), such that 2S+1 equals the multiplicity.
        rdm1:   Spin-free one-particle density matrix.
        rdm2:   Spin-free two-particle density matrix in PySCF convention.

    Returns:
        alpha-1-rdm, beta-1-rdm

    Raises:
        ValueError: invalid input
    """
    if N < 1:
        raise ValueError('The number of electrons must be positive.')
    if S < 0:
        raise ValueError('The spin must be positive.')

    # Calculate the spin density (rdm1a - rdm2b)
    spindens_term1 = (2 - 0.5 * N) / (S + 1) * rdm1
    spindens_term2 = 1 / (S + 1) * np.einsum('pkkq->pq', rdm2)
    spindens = spindens_term1 - spindens_term2

    # Calculate rdm1a and rdm1b using the total density (rdm1a + rdm1b) and the spin density.
    rdm1a = 0.5 * (rdm1 + spindens)
    rdm1b = 0.5 * (rdm1 - spindens)
    return rdm1a, rdm1b
