# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Simple wrapper functions to find active spaces (semi-)automatically."""

from typing import Union, Tuple, List, Dict, Set, Any
from enum import Enum
import numpy as np

from pyscf.gto import Mole
from pyscf.scf import UHF
from pyscf.scf.hf import SCF as SCFclass
from pyscf.scf.hf import RHF as RHFclass
from pyscf.scf.rohf import ROHF as ROHFclass
from pyscf.scf.uhf import UHF as UHFclass
from pyscf.dft.rks import KohnShamDFT

from asf.natorbs import MP2NaturalOrbitals, selectNaturalOccupations
from asf.asf import default_entropy_threshold, default_plateau_threshold, ASFBase, calc_ncore, \
    inactive_orbital_lists
from asf import ASFCI, ASFDMRG


# Attempt to import native DF-MP2 classes. If the PySCF version does not have those, use the
# "normal" MP2 code instead.
try:
    from pyscf.mp.dfmp2_native import DFRMP2
    from pyscf.mp.dfump2_native import DFUMP2
except ImportError:
    from pyscf.mp import MP2
    have_native_dfmp2 = False
else:
    have_native_dfmp2 = True


def loghead(text: str, verbose: bool) -> None:
    """
    Print an important message with hyphens and space above and below.

    Args:
        text:       text to be printed
        verbose:    only print if True
    """
    if verbose:
        print('')
        print('-' * 80)
        print(text)
        print('-' * 80)
        print('')


def loginfo(text: str, verbose: bool) -> None:
    """
    Print an informational message.

    Args:
        text:       text to be printed
        verbose:    only print if True
    """
    if verbose:
        print(text)


# Be orderly about the SCF types we can encounter.
SCFtype = Enum('SCFtype', 'RHF UHF ROHF')


def mp2_from_scf(mf: Union[RHFclass, UHFclass],
                 lower: float = 0.01,
                 upper: float = 1.99,
                 min_orb: int = 10,
                 max_orb: int = 30) -> \
        Tuple[int, np.ndarray, np.ndarray]:
    """
    Calculate MP2 natural orbitals and select an orbital subset.

    Given a converged Hartree-Fock solution, the function calculates MP2 natural orbitals (from the
    orbital-unrelaxed density matrix). An active space is selected by including all natural
    orbitals with associated eigenvalues between a lower and an upper threshold. In addition,
    boundaries are imposed to select no less than a minimum and no more than a maximum number of
    orbitals for the active space.

    Args:
        mf:         RHF or UHF.
        lower:      Lower threshold for orbital selection based on MP2 natural occupation numbers.
        upper:      Upper threshold for orbital selection based on MP2 natural occupation numbers.
        min_orb:    Minimum number of orbitals to select, if possible.
        max_orb:    Maximum number of orbitals to select in any case.

    Returns:
        Tuple containing (no. of act. electrons, list of act. orbitals, MO coefficients)

    Raises:
        TypeError:      Argument with invalid type.
        RuntimeError:   Unexpected internal error.
    """
    # Identify the type of the SCF object.
    scf_type = None
    if isinstance(mf, RHFclass):
        scf_type = SCFtype.RHF
    if isinstance(mf, ROHFclass):
        scf_type = SCFtype.ROHF
    if isinstance(mf, UHFclass):
        scf_type = SCFtype.UHF

    if scf_type not in (SCFtype.RHF, SCFtype.UHF):
        raise TypeError('mf must be a RHF or UHF object.')

    if isinstance(mf, KohnShamDFT):
        raise TypeError('DFT objects are not supported.')

    # Calculate natural orbitals and natural occupation numbers for MP2.
    if have_native_dfmp2:
        # Use the RHF-DF-MP2 or UHF-DF-MP2 implementation if available.
        if scf_type == SCFtype.RHF:
            natocc, natorb = DFRMP2(mf).make_natorbs()
        elif scf_type == SCFtype.UHF:
            natocc, natorb = DFUMP2(mf).make_natorbs()
    else:
        # Otherwise, there is an older DFMP2 code we can use for RHF.
        if scf_type == SCFtype.RHF:
            pt = MP2(mf).density_fit()
        # Only conventional MP2 is available for UHF in that case.
        elif scf_type == SCFtype.UHF:
            pt = MP2(mf)
        pt.kernel()
        natocc, natorb = MP2NaturalOrbitals(pt)

    # Ascertain that the natural occupation numbers are in descending order.
    if np.any(np.diff(natocc > 0.0)):
        raise RuntimeError('Natural occupation numbers are not in descending order.')

    # An active space is selected using the MP2 natural occupation numbers.
    nel, mo_list = selectNaturalOccupations(natocc, lower, upper, max_orb, min_orb)

    # Return number of active electrons, list of active orbitals, MO coefficients.
    return nel, mo_list, natorb


def reorder_mos(mol: Mole,
                nel: int,
                mo_list: List[int],
                mo_coeff: np.ndarray) -> \
        Tuple[List[int], np.ndarray]:
    """
    Arrange MO columns in the order occupied/core, active, virtual from left to right.

    This function is analogous to sort_mo(...) in pyscf.mcscf.

    Args:
        mol:        Mole object.
        nel:        Number of electrons in the active space.
        mo_list:    List of active orbitals.
        mo_coeff:   MO coefficients. Orbitals that are not in mo_list must be ordered such that
                    occupied orbitals come first and virtual orbitals second (from left to right).

    Returns:
        Tuple (new list of active orbitals, reordered MO coefficients)
    """
    # number of core, active and all MOs
    ncore = calc_ncore(mol, nel)
    nact = len(mo_list)
    nmo = mo_coeff.shape[1]

    # current lists of core and virtual orbitals.
    occ_list, virt_list = inactive_orbital_lists(ncore, nmo, mo_list)

    # Crop the respective subsets of MO coefficients.
    mo_occ = mo_coeff[:, occ_list]
    mo_act = mo_coeff[:, mo_list]
    mo_virt = mo_coeff[:, virt_list]

    # Put the MO coefficients back together.
    mo_sorted = np.hstack((mo_occ, mo_act, mo_virt))

    # State the obvious in the MO list.
    act_sorted = list(range(ncore, ncore + nact))

    return act_sorted, mo_sorted


def merge_active_spaces(mol: Mole,
                        nel1: int,
                        mo_list1: List[int],
                        nel2: int,
                        mo_list2: List[int]) -> \
        Tuple[int, List[int]]:
    """
    Perform a simple merging of active spaces.

    This function attempts to construct an active space for state-averaged CASSCF calculations, by
    forming the union of the lists of active orbitals for individual electronic states.

    This function is based on the following assumptions:
        - A sensible active space can be obtained simply by merging mo_list1 and mo_list2.
        - Orbitals that are not already active in both active space definitions "bring" either two
          electrons (occupied MO) or no electrons (virtual MO) into the merged active space.
        - Inactive MOs are always ordered: doubly occupied orbitals come before virtual orbitals.

    Args:
        mol:        Mole object with molecular data.
        nel1:       Number of electrons in the first active space.
        mo_list1:   List of orbitals in the first active space.
        nel2:       number of electrons in the second active space.
        mo_list2:   List of orbitals in the second active space.

    Returns:
        Tuple representing the merged active space: (number of electrons, list of orbitals)

    Raises:
        ValueError: Insonsistent input provided.
    """
    mo_set1 = set(mo_list1)
    mo_set2 = set(mo_list2)

    if len(mo_set1) != len(mo_list1) or len(mo_set2) != len(mo_list2):
        raise ValueError('Duplicate elements in lists of active orbitals.')

    if nel1 > 2 * len(mo_set1) or nel2 > 2 * len(mo_set2):
        raise ValueError('Too many electrons to be accomodated in the active orbitals.')

    # The merged set of active spaces.
    mo_merged = set.union(mo_set1, mo_set2)

    # Number of occupied ("core") orbitals associated with each active space.
    ncore1 = calc_ncore(mol, nel1)
    ncore2 = calc_ncore(mol, nel2)

    # Set of core orbitals associated with the first active space.
    core_set1: Set[int] = set()
    ctr = 0
    while len(core_set1) < ncore1:
        if ctr not in mo_set1:
            core_set1.add(ctr)
        ctr += 1

    # Set of core orbitals associated with the second active space.
    core_set2: Set[int] = set()
    ctr = 0
    while len(core_set2) < ncore2:
        if ctr not in mo_set2:
            core_set2.add(ctr)
        ctr += 1

    # Determine the number of electrons in the merged active space.
    nel = nel1 + 2 * len(core_set1.intersection(mo_merged))

    # Check consistency of the number of electrons.
    if nel2 + 2 * len(core_set2.intersection(mo_merged)) != nel:
        raise ValueError('Active space specifications are inconsistent.')

    return nel, sorted(mo_merged)


def do_ci(mol: Mole,
          nel_mp2: int,
          molist_mp2: np.ndarray,
          natorb_mp2: np.ndarray,
          spin: int = 0,
          nroots: int = 1,
          tol: float = 1.0e-6,
          maxM: int = 250,
          switch_dmrg: int = 12,
          dmrg_kwargs: Dict[str, Any] = None,
          fci_kwargs: Dict[str, Any] = None) -> ASFBase:
    """
    Perform the DMRGCI/CASCI calculation using the selected set of natural orbitals.

    If the initial active space is small, use the normal FCI solver instead.

    Args:
        mol:                Mole instance containing the molecular data.
        nel_mp2:            Number of electrons of active space selected by MP2.
        molist_mp2:         List of selected orbital by MP2 calculations.
        natorb_mp2:         MP2 Natural orbitals used for orbital selection.
        spin:               Spin of the requested CASCI/DMRGCI calculation.
        nroots:             Number of CI roots (excited states, 1=groundstate) to calculate.
        tol:                Convergence tolerance for DMRG.
        switch_dmrg:        Number of orbitals to switch between the regular FCI solver and DMRG.
        maxM:               Maximal DMRG bond dimension.
        dmrg_kwargs:        Further keyword arguments to be passed to the DMRG fcisolver.
        fci_kwargs:         Further keyword arguments to be passed to CASCI.fcisolver.

    Returns:
        calc:               CI results.
    """
    if dmrg_kwargs is None:
        dmrg_kwargs = {}
    if fci_kwargs is None:
        fci_kwargs = {}
    calc: ASFBase
    if len(molist_mp2) > switch_dmrg:
        # Ugly, but we need to do this now, because DMRGCI.spin is not used correctly...
        mol_copy = mol.copy()
        mol_copy.spin = spin
        mol_copy.build()
        # Now perform the DMRG calculation with the desired spin.
        calc_dmrg = ASFDMRG(mol_copy, natorb_mp2, nel=nel_mp2, mo_list=molist_mp2)
        calc_dmrg.runDMRG(nroots=nroots, maxM=maxM, tol=tol, **dmrg_kwargs)
        calc = calc_dmrg
    else:
        calc_ci = ASFCI(mol, natorb_mp2, nel_mp2, mo_list=molist_mp2)
        calc_ci.runCI(nroots=nroots, spin=spin, **fci_kwargs)
        calc = calc_ci
    return calc


def runasf_from_scf(mf: SCFclass,
                    entropy_threshold: float = default_entropy_threshold,
                    plateau_threshold: float = default_plateau_threshold,
                    states: Union[int, List[Tuple[int, int]]] = None,
                    sort_mos: bool = False,
                    maxM: int = 250,
                    tol: float = 1.0e-6,
                    mp2_kwargs: Dict[str, Any] = None,
                    switch_dmrg: int = 12,
                    dmrg_kwargs: Dict[str, Any] = None,
                    fci_kwargs: Dict[str, Any] = None,
                    verbose: bool = True) -> \
        Tuple[int, List[int], np.ndarray]:
    """
    Entropy-based active space suggestion with DMRG from a set of MP2 natural orbitals.

    The function executes the following steps:
        1)  Calculate MP2 natural orbitals using the SCF object provided in the input.
        2)  Select an initial set of MP2 natural orbitals based on their eigenvalues.
        3)  Perform a DMRG-CASCI calculation using the previously determined orbital subset.
        4)  Select the final active space based on the entropies from the DMRG calculation.

    The output of the function contains:
        - the number of active electrons,
        - the list of MP2 natural orbitals selected for the active space,
        - and the MO coefficient matrix containing all (active and inactive) MP2 natural orbitals.

    The columns representing the inactive orbitals in the MO coefficient matrix are ordered
    according to the convention of PySCF: occupied MOs to the left and virtual MOs to the right.
    By default, the original ordering of the MP2 natural orbitals is preserved. This can be
    changed by setting reorder_mos to True: then, the columns of the MO coefficients are reordered
    such that they can be passed directly to CASCI/CASSCF without any further sorting.

    This function does not provide active spaces for individual excited electronic states.
    However, it can be used to obtain active space suggestions for state-averaged calculations.
    If a value is provided for 'states', the function performs active space selections for all spin
    and electronic states, and returns a representation of the combined active space.

    Args:
        mf:                 RHF or UHF object containing converged orbitals.
        entropy_threshold:  Single-orbital entropy cutoff for active space selection.
        plateau_threshold:  Cutoff for "plateaus" of orbitals with similar entropies.
        states:             Electronic states to construct the active space for.
                            None (default): Ground state for the same spin as in mf.
                            Integer n: lowest n electronic states, same spin as in mf.
                            List of Tuples [(s, n)]: lowest n electronic states for each spin s.
        sort_mos:           If True, reorder the MO columns as occupied, active, virtual.
        maxM:               Maximal DMRG bond dimension.
        tol:                Convergence tolerance for DMRG.
        mp2_kwargs:         Keyword arguments for MP2 orbital selection (see mp2_from_scf(...)).
        switch_dmrg:        Number of orbitals to switch between the regular FCI solver and DMRG.
        dmrg_kwargs:        Further keyword arguments to be passed to the DMRG fcisolver.
        fci_kwargs:         Further keyword arguments to be passed to CASCI.fcisolver.
        verbose:            Print information if set to True.

    Returns:
        Tuple with (no. of act. electrons, list of act. orbitals, orbital coefficients)
    """
    # Avoiding trouble with mutable default arguments...
    if mp2_kwargs is None:
        mp2_kwargs = {}
    if dmrg_kwargs is None:
        dmrg_kwargs = {}
    if fci_kwargs is None:
        fci_kwargs = {}
    mol = mf.mol
    if states is None:
        spin_roots = [(mol.spin, 1)]
    elif isinstance(states, int):
        spin_roots = [(mol.spin, states)]
    else:
        spin_roots = [(spin, nroots) for (spin, nroots) in states]

    # Calculate MP2 natural orbitals from the SCF object.
    # Select an initial set of the resulting MOs for the subsequent calculation.
    loghead('Calculating MP2 natural orbitals', verbose)
    nel_mp2, molist_mp2, natorb_mp2 = mp2_from_scf(mf, **mp2_kwargs)
    loginfo('', verbose)
    loginfo('-> Selected initial orbital window of {0:d} electrons in {1:d} MP2 natural orbitals.'.
            format(nel_mp2, len(molist_mp2)), verbose)

    # Determine active spaces for all spin states and electronic states.
    for iteration, (spin, nroots) in enumerate(spin_roots):

        loghead('Calculating single-orbital entropies for the initial active space', verbose)
        loginfo('spin = {0:d}, number of roots = {1:d}'.format(spin, nroots), verbose)
        loginfo('', verbose)

        calc = do_ci(mol=mol, nel_mp2=nel_mp2, molist_mp2=molist_mp2, natorb_mp2=natorb_mp2,
                     spin=spin, nroots=nroots, tol=tol, maxM=maxM, switch_dmrg=switch_dmrg,
                     dmrg_kwargs=dmrg_kwargs, fci_kwargs=fci_kwargs)

        # Perform the entropy-based orbital selection.
        for root in range(nroots):
            loginfo('', verbose)
            loginfo('Orbital selection for state {0:d}:'.format(root), verbose)
            loginfo('', verbose)

            nel_state, mo_list_state = \
                calc.selectOrbitals(root=root, threshold=entropy_threshold,
                                    plateau_threshold=plateau_threshold, verbose=verbose)

            # Merge current active space with the already existing space.
            if iteration == 0:
                nel, mo_list = nel_state, mo_list_state
            else:
                nel, mo_list = merge_active_spaces(mol, nel, mo_list, nel_state, mo_list_state)

    loghead('Active space selection finished', verbose)
    loginfo('-> Selected an active space of {0:d} electrons in {1:d} orbitals.'.
            format(nel, len(mo_list)), verbose)

    # Return the number of active electrons, active MO indices, MP2 natural orbital coefficients.
    if sort_mos:
        act_sorted, mo_sorted = reorder_mos(mol, nel, mo_list, natorb_mp2)
        return nel, act_sorted, mo_sorted
    else:
        return nel, mo_list, natorb_mp2


default_scf_settings = {
    'max_cycle': 100,
    'conv_tol': 1.0e-7,
    'level_shift': 0.2,
    'damp': 0.2
}


class FailedSCF(Exception):
    """Raised if the SCF calculation failed to converge."""

    pass


def runasf_from_mole(mol: Mole,
                     scf_kwargs: Dict[str, Any] = None,
                     stability_analysis: bool = True,
                     verbose: bool = True,
                     **kwargs) -> \
        Tuple[int, List[int], np.ndarray]:
    """
    Perform SCF calculation and suggest an active space for a molecule.

    This function attempts to calculate a stable UHF solution for the Mole object provided.
    Subsequently, runasf_from_scf(...) is called to calculate MP2 natural orbitals, and to select
    an active space from those.

    For more details, please refer to runasf_from_scf(...).

    Args:
        mol:                Mole instance containing the molecular data.
        scf_kwargs:         Dictionary of options to be set for the SCF object.
                            Note: this function deviates from PySCF defaults for a few settings.
        stability_analysis: Perform a stability analysis of the SCF solution if True.
                            Re-start the SCF calculation once of the solution is unstable.
        verbose:            Print information if set to True.
        **kwargs:           Keyword arguments to be passed to runasf_from_scf(...).

    Returns:
        Active space suggestion as provided by runasf_from_scf(...).

    Raises:
        FailedSCF: SCF calculation failed to converge or is unstable.
    """
    # Create a copy of the dictionary with the SCF settings. Include any entries from scf_kwargs.
    # Values provided in scf_kwargs always take priority over the defaults.
    scf_settings = default_scf_settings.copy()
    if scf_kwargs is not None:
        scf_settings.update(scf_kwargs)

    # Perform the UHF calculation.
    loghead("Calculating UHF orbitals", verbose)
    mf = UHF(mol)
    for key, value in scf_settings.items():
        setattr(mf, key, value)
    mf.kernel()

    # Stability analysis, if requested.
    if stability_analysis:

        # The stability analysis method does not provide a flag to indicate if the solution is
        # stable. Instead, check if the new orbital suggestion is identical with the SCF MOs.
        loginfo('', verbose)
        loginfo('Performing stability analysis.', verbose)
        mo_restart = mf.stability()[0]
        if not np.allclose(mo_restart, mf.mo_coeff, atol=1e-12, rtol=0.0):

            loginfo('', verbose)
            loginfo('SCF solution unstable.', verbose)
            loginfo('Restarting the SCF calculation.', verbose)
            loginfo('', verbose)

            # Re-start SCF with the MOs from the stability analysis as the new guess.
            dm_restart = mf.make_rdm1(mo_coeff=mo_restart)
            mf.kernel(dm0=dm_restart)

            # Check once again if the solution is stable.
            loginfo('', verbose)
            loginfo('Performing stability analysis.', verbose)
            mo_restart = mf.stability()[0]
            if not np.allclose(mo_restart, mf.mo_coeff, atol=1e-12, rtol=0.0):
                raise FailedSCF('SCF solution is not stable.')

    # Always check if the SCF procedure has converged.
    if not mf.converged:
        raise FailedSCF('SCF did not converge.')

    loginfo('', verbose)
    loginfo('-> SCF solution converged and stable.', verbose)

    # Finally, calculate an active space suggestion.
    return runasf_from_scf(mf, verbose=verbose, **kwargs)
