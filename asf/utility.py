# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Utility functions."""

import os
import re
import subprocess  # nosec B404
import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import which
from typing import Generator, Optional, Sequence, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
from pyscf.gto import Mole
from pyscf.lib import parameters
from pyscf.mcscf.casci import CASCI as CASCIclass
from pyscf.tools import molden

ENTROPY_ZERO_CUTOFF = 1.0e-12
""" Default value to avoid calculating the logarithm of negative numbers."""


@contextmanager
def tempname(
    directory: str = parameters.TMPDIR, suffix: Optional[str] = None
) -> Generator[str, None, None]:
    """Context manager that creates a temporary file name, and deletes the file at exit.

    In contrast to tempfile.TemporaryFile, it only returns the name and not a file object.
    This permits the user to hand it over to functions which expect a file name instead of a file
    object, hand it over to other programs for opening and closing, etc.

    Args:
        directory:    Directory to store the temporary file.
        suffix: Suffix for the temporary file name.

    Yields:
        The name of the temporary file.
    """
    # First we create the temporary file on disk.
    _, filename = tempfile.mkstemp(dir=directory, suffix=suffix)
    try:
        # Provides the file name in "with tempname(...) as ...:".
        yield filename
    finally:
        # Make sure to clean up in the end.
        os.remove(filename)


def write_jmol_script(
    stream: TextIO,
    mo_filename: Union[Path, str],
    mo_list: Union[list[int], np.ndarray],
    mo_titleformat: str = "MO %I of %N",
    name: str = "mo",
    cutoff: float = 0.04,
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    zoom: int = 60,
    script_options: str = "",
) -> None:
    """Write a Jmol script for creating pictures of orbitals to an IO stream.

    Note that the indices of orbitals in mo_list start from zero, whereas Jmol counts orbitals
    starting from one.

    Args:
        stream:         IO stream to which the contents of the script is written.
        mo_filename:    Path-like filename for file containing MOs to plot
        mo_list:        Indices of MOs to be plotted (indexing based on zero).
        mo_titleformat: A Jmol template string for MO titles.
        name:           File names of pictures will be: [name]_[MO index].png
        cutoff:         Isosurface cutoff for orbital rendering.
        rotate:         Rotation angle about the (x-axis, y-axis, z-axis) in degrees.
        zoom:           Zooming percentage for Jmol. Factor 100 is suitable to show all atoms, but
                        isosurfaces may appear cropped.
        script_options: A string with arbitrary input to be included in the Jmol script.
    """
    stream.write(f'load "{mo_filename!s}"\n')
    # White background, no "Jmol" label in the picture.
    stream.write("background white\n")
    stream.write("frank off\n")
    # Represent molecule as sticks with radius 0.1 Angstrom.
    stream.write("spacefill off\n")
    stream.write("wireframe 0.1\n")
    # Settings to render MOs.
    stream.write("mo fill nomesh translucent\n")
    stream.write("mo color yellow purple\n")
    stream.write("mo resolution 20\n")
    stream.write(f"mo cutoff {cutoff:f}\n")
    # Labelling in the picture.
    stream.write(f'mo titleformat "{mo_titleformat}"\n')
    # Rotate the molecule into a different orientation.
    stream.write(f"rotate {rotate[0]:f} x\n")
    stream.write(f"rotate {rotate[1]:f} y\n")
    stream.write(f"rotate {rotate[2]:f} z\n")
    stream.write(f"zoom {zoom:d}\n")
    # Give the user a chance to override any settings.
    if script_options:
        stream.write(script_options + "\n")
    # Now iterate over the MOs and dump PNG files.
    for i in mo_list:
        stream.write(f"mo {i+1:d}\n")
        stream.write(f'write image png "{name}_{i:d}.png"\n')


def pictures_Jmol(
    mol: Mole,
    mo_coeff: np.ndarray,
    mo_list: Union[list[int], np.ndarray],
    name: str = "mo",
    cutoff: float = 0.04,
    mo_index_title: str = "MO %I of %N",
    ene: Optional[Union[np.ndarray, list[float]]] = None,
    occ: Optional[Union[np.ndarray, list[float]]] = None,
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    zoom: int = 60,
    script_options: str = "",
    jmol_exec: str = "jmol",
    jmol_opts: str = "--nodisplay",
    xvfb_cmd: Optional[str] = None,
    suppress_out: bool = True,
) -> None:
    """Create pictures of orbitals using Jmol.

    Note that the indices of orbitals in mo_list start from zero, whereas Jmol counts orbitals
    starting from one.

    Args:
        mol:            The molecule as an instance of pyscf.gto.Mole.
        mo_coeff:       Matrix of MO coefficients.
        mo_list:        Indices of MOs to be plotted (indexing based on zero).
        name:           File names of pictures will be: [name]_[MO index].png
        mo_index_title: Title format for showing orbital numbers.
        cutoff:         Isosurface cutoff for orbital rendering.
        ene:            If provided, orbital energies will included in the MO titles.
        occ:            If provided, orbital occupancies will be included in the MO titles.
        rotate:         Rotation angle about the (x-axis, y-axis, z-axis) in degrees.
        zoom:           Zooming percentage for Jmol. Factor 100 is suitable to show all atoms, but
                        isosurfaces may appear cropped.
        script_options: A string with arbitrary input to be included in the Jmol script.
        jmol_exec:      Name of the Jmol executable.
        jmol_opts:      String containing to be provided to Jmol.
        xvfb_cmd:       X virtual framebuffer command.
                        None (default): Use 'xvfb-run' if it is available.
                        string:         Enforces the supplied name for the xfvb-run command.
                        '':             Call Jmol without xvfb-run.
        suppress_out:   Suppress stdout and stderr output of Jmol if set to True.
    """
    # Set xvfb option.
    if xvfb_cmd is None:
        xvfb_cmd = "xvfb-run" if which("xvfb-run") else ""

    # Apparently, Jmol does not work properly if the script does not have an .spt ending.
    with tempname(suffix=".molden") as molden_filename, tempname(suffix=".spt") as jmol_filename:
        # Dump the orbitals into the temporary molden file.
        molden.from_mo(mol, molden_filename, mo_coeff, ene=ene, occ=occ)

        mo_titleoptions = []
        if mo_index_title:
            mo_titleoptions.append(mo_index_title)
        if mol.symmetry:
            mo_titleoptions.append("Symmetry: %S")
        if ene is not None:
            mo_titleoptions.append("Energy: %.4E %U")
        if occ is not None:
            mo_titleoptions.append("Occupancy: %O")
        mo_titleformat = "|".join(mo_titleoptions)

        # Open Jmol script file (temporary) for writing.
        with open(jmol_filename, "w") as file_jmol:
            write_jmol_script(
                stream=file_jmol,
                mo_list=mo_list,
                mo_filename=molden_filename,
                mo_titleformat=mo_titleformat,
                name=name,
                cutoff=cutoff,
                rotate=rotate,
                zoom=zoom,
                script_options=script_options,
            )

        # Execute Jmol using the files we have created previously.
        if xvfb_cmd:
            runargs = [xvfb_cmd, jmol_exec, jmol_opts, jmol_filename]
        else:
            runargs = [jmol_exec, jmol_opts, jmol_filename]

        if suppress_out:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = subprocess.STDOUT
            stderr = subprocess.STDOUT

        subprocess.run(
            runargs,
            check=False,
            shell=False,  # noqa: S603 # nosec B603
            stdout=stdout,
            stderr=stderr,
        )


def corresponding_orbitals(
    mol: Mole, mo_coeff1: np.ndarray, mo_coeff2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform corresponding orbital transformation for two orbital sets.

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
    ovlp_ao = mol.intor_symmetric("int1e_ovlp")

    # Overlap of the two MO sets.
    ovlp_mo = np.linalg.multi_dot([mo_coeff1.T, ovlp_ao, mo_coeff2])

    # SVD of the MO overlap.
    U, sigma, Vt = np.linalg.svd(ovlp_mo)

    # The corresponding orbitals.
    co_coeff1 = mo_coeff1.dot(U)
    co_coeff2 = mo_coeff2.dot(Vt.T)

    return sigma, co_coeff1, co_coeff2


def compare_active_spaces(
    cas: CASCIclass,
    mo_guess: np.ndarray,
    mo_list: Union[Sequence[int], np.ndarray],
    full_result: bool = False,
) -> Union[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Given a CASSCF (or CASCI) object, calculate its consistency with some guess orbitals.

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
    final_active = cas.mo_coeff[:, ncore : ncore + ncas]

    # Perform corresponding orbital transformation of the active spaces.
    sigma, co_guess, co_cas = corresponding_orbitals(mol, guess_active, final_active)

    if full_result:
        return (sigma, co_guess, co_cas)
    else:
        return float(min(sigma))


def rdm1s_from_rdm12(
    N: int, S: float, rdm1: np.ndarray, rdm2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the alpha and beta components of the 1-RDM from the spin-free 1-RDM and 2-RDM.

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
        raise ValueError("The number of electrons must be positive.")
    if S < 0:
        raise ValueError("The spin must be positive.")

    # Calculate the spin density (rdm1a - rdm2b)
    spindens_term1 = (2 - 0.5 * N) / (S + 1) * rdm1
    spindens_term2 = 1 / (S + 1) * np.einsum("pkkq->pq", rdm2)
    spindens = spindens_term1 - spindens_term2

    # Calculate rdm1a and rdm1b using the total density (rdm1a + rdm1b) and the spin density.
    rdm1a = 0.5 * (rdm1 + spindens)
    rdm1b = 0.5 * (rdm1 - spindens)
    return rdm1a, rdm1b


def transform_dm(dm: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Perform an orthogonal basis transformation of a density matrix.

    Args:
        dm: The density matrix to be transformed.
        U:  An orthogonal transformation matrix.
            First index / rows: the old basis
            Second index / columns: the new basis

    Returns:
        The transformed density matrix.
    """
    # Transforming the indices:
    # Contract the first index of the density matrix with U. After applying tensordot, the
    # transformed index ends up rightmost, and the second index becomes the first index.
    # This way, we cycle through all indices and obtain the original order in the end.
    dm_transformed = dm.copy()
    for _ in range(dm.ndim):
        dm_transformed = np.tensordot(dm_transformed, U, axes=(0, 0))
    return dm_transformed


def iround(x: float) -> int:
    """A safe variant of rounding that always returns an int.

    Python's round can return float, e.g. if its argument is a numpy.float.

    Args:
        x:                      Floating point number to be rounded

    Returns:
        rounded number
    """
    if x >= 0.0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)


def calculate_entropy_s1(
    orbdens: np.ndarray, zero_cutoff: float = ENTROPY_ZERO_CUTOFF
) -> np.ndarray:
    """Calculates the single-site entropies from the one-orbital density.

    Args:
        orbdens:                one-orbital densities in an N x 4 array
        zero_cutoff:            Threshold to determine values that are effectively zero

    Returns:
        single-site entropies

    Raises:
        Exception: various errors
    """
    if not ((orbdens.ndim == 2) and (orbdens.shape[1] == 4)):
        raise Exception("Orbital Density has wrong dimensions.")
    if not (zero_cutoff >= 0.0):
        raise Exception("Zero cut off should be larger than 0.")

    # Locations where the one-orbital density is positive
    where_positive = orbdens > zero_cutoff
    # Locations where the one-orbital density is zero (within the tolerance)
    where_zero = np.logical_and(orbdens <= zero_cutoff, orbdens >= -zero_cutoff)
    # Ensure there are no negative entries
    if not (np.all(orbdens >= -zero_cutoff)):
        raise Exception("Negative values in orb density.")

    # Now calculate the actual entropies.
    WlogW = orbdens * np.log(orbdens, where=where_positive)
    WlogW[where_zero] = 0.0
    s1 = -np.sum(WlogW, axis=1)
    return s1


def orbdens_from_rdm(rdm1a: np.ndarray, rdm1b: np.ndarray, rdm2ab: np.ndarray) -> np.ndarray:
    """Calculate the one-orbital density from the reduced density matrices.

    Args:
        rdm1a:  Spin-up one-particle reduced density matrix.
        rdm1b:  Spin-down one-particle reduced density matrix.
        rdm2ab: Mixed-spin component of the two-particle density matrix.
                OR: 0.5 * the spin-free two-particle density matrix.

    Returns:
        One-orbital density. Columns: empty, spin-up, spin-down, doubly occupied.

    Raises:
        ValueError: Invalid input
    """
    nmo = rdm1a.shape[0]
    if rdm1a.shape != (nmo, nmo):
        raise ValueError("Shape of rdm1a must be (nmo, nmo).")
    if rdm1b.shape != (nmo, nmo):
        raise ValueError("Shape of rdm1b must be (nmo, nmo).")
    if rdm2ab.shape != (nmo, nmo, nmo, nmo):
        raise ValueError("Shape of rdm2 must be (nmo, nmo, nmo, nmo).")
    orbdens = np.zeros((nmo, 4))
    for p in range(nmo):
        orbdens[p, 0] = 1.0 - rdm1a[p, p] - rdm1b[p, p] + rdm2ab[p, p, p, p]
        orbdens[p, 1] = rdm1a[p, p] - rdm2ab[p, p, p, p]
        orbdens[p, 2] = rdm1b[p, p] - rdm2ab[p, p, p, p]
        orbdens[p, 3] = rdm2ab[p, p, p, p]
    return orbdens


def loghead(text: str, verbose: bool) -> None:
    """Print an important message with hyphens and space above and below.

    Args:
        text:       text to be printed
        verbose:    only print if True
    """
    if verbose:
        print("")
        print("-" * 80)
        print(text)
        print("-" * 80)
        print("")


def loginfo(text: str, verbose: bool) -> None:
    """Print an informational message.

    Args:
        text:       text to be printed
        verbose:    only print if True
    """
    if verbose:
        print(text)


def show_mos_grid(
    images: list[Path],
    mo_list: Optional[list[int]] = None,
    columns: int = 5,
    figsize: tuple[int, int] = (14, 14),
    image_regex: str = r"mo_(\d+)",
    **fig_kwargs,
) -> None:
    """Show molecular orbitals images in a grid.

    The function assumes that the MO images follow a systematic filename convention containing the
    molecular orbital index. By default, the same format as in `pictures_Jmol` is used, i.e.
    `mo_[idx].png`, where `[idx]` is the respective MO index (indexing starting from 0).

    Args:
        images:         List of image paths.
        mo_list:        MO indices to show.
        columns:        Number of grid columns.
        figsize:        Matplotlib figure size.
        image_regex:    Regular expression matching the index in the orbital image filename. Custom
                        expressions should be written, so that the index is matched in a group.
        fig_kwargs:     Extra arguments that are forwared to `matplotlib.pyplot.subplots()`.
    """

    def get_index(img_path: Path) -> Optional[int]:
        """Infer orbital index from the filename.

        Args:
            img_path:  Image path.

        Returns:
            Orbital index if regular expression match is successful.
        """
        match = re.match(image_regex, img_path.name)
        if match:
            return int(match.group(1))
        return None

    n_img = len(images) if mo_list is None else len(mo_list)
    rows = (n_img // columns) + 1

    to_plot = images
    if mo_list is not None:
        to_plot = [p for p in images for i in mo_list if get_index(p) == i]

    _, axs = plt.subplots(rows, columns, figsize=figsize, layout="compressed", **fig_kwargs)
    axs = axs.reshape((rows, columns))
    row = 0
    for index, img_path in enumerate(to_plot):
        mo = plt.imread(img_path)
        mo_index = get_index(img_path)
        col = index % columns
        axs[row, col].imshow(mo)
        axs[row, col].axis("off")
        if mo_index is not None:
            axs[row, col].text(0.5, 0.5, f"MO {mo_index}")
        if col == columns - 1:
            row += 1

    for r in range(rows):
        for c in range(columns):
            if (r * columns) + c >= n_img:
                axs[r, c].set_visible(False)
    plt.show()
