# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Simple wrapper functions to find active spaces (semi-)automatically."""

from typing import Any, Optional, Union

import numpy as np
from pyscf.gto import Mole
from pyscf.scf.hf import SCF as SCFclass

from .asfbase import (
    DEFAULT_ENTROPY_THRESHOLD,
    ActiveSpace,
    ASFBase,
    FilterFunction,
    calc_ncore,
    inactive_orbital_lists,
    merge_active_spaces,
    print_mo_table,
)
from .casci import ASFCI
from .dmrg import ASFDMRG
from .pairinfo import MOListInfo
from .preselection import MP2NatorbPreselection
from .scf import DEFAULT_MAX_RESTARTS, DEFAULT_SCF_SETTINGS, stable_scf
from .utility import loghead, loginfo


def reorder_mos(
    mol: Mole, nel: int, mo_list: list[int], mo_coeff: np.ndarray
) -> tuple[list[int], np.ndarray]:
    """Arrange MO columns in the order occupied/core, active, virtual from left to right.

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


def create_asf_switched_cisolver(
    mol: Mole,
    initial_space: ActiveSpace,
    spin: int = 0,
    nroots: int = 1,
    switch_dmrg: int = 12,
    dmrg_kwargs: Optional[dict[str, Any]] = None,
    fci_kwargs: Optional[dict[str, Any]] = None,
) -> ASFBase:
    """Create an ASFCI or ASFDMRG object depending on the number of active orbitals.

    This function sets up an ASF object that manages either a CASCI or DMRGCI calculation. By
    default an `ASFCI` object is created, but if the number of active orbitals surpasses the value
    of `switch_dmrg`, an `ASFDMRG` object is created instead.

    Args:
        mol:                Mole instance containing the molecular data.
        initial_space:      Initial space to perform search for final active space.
        spin:               Spin of the requested CASCI/DMRGCI calculation.
        nroots:             Number of CI roots (excited states, 1=groundstate) to calculate.
        switch_dmrg:        Number of orbitals to switch between the regular FCI solver and DMRG.
        dmrg_kwargs:        Further keyword arguments to be passed to the DMRG fcisolver.
        fci_kwargs:         Further keyword arguments to be passed to CASCI.fcisolver.

    Returns:
        ASF object
    """
    if dmrg_kwargs is None:
        dmrg_kwargs = {}
    if fci_kwargs is None:
        fci_kwargs = {}

    _mol = mol
    if spin != mol.spin:
        _mol = mol.copy()
        _mol.spin = spin
        _mol.build()

    calc: ASFBase
    if initial_space.norb > switch_dmrg:
        # Now perform the DMRG calculation with the desired spin.
        calc = ASFDMRG(
            _mol,
            initial_space.mo_coeff,
            nel=initial_space.nel,
            mo_list=initial_space.mo_list,
            nroots=nroots,
            maxM=dmrg_kwargs.get("maxM", 250),
            tol=dmrg_kwargs.get("tol", 1.0e-6),
            fcisolver_kwargs=dmrg_kwargs,
        )
    else:
        fci_kwargs.update(spin=spin)
        calc = ASFCI(
            _mol,
            initial_space.mo_coeff,
            nel=initial_space.nel,
            mo_list=initial_space.mo_list,
            nroots=nroots,
            fcisolver_kwargs=fci_kwargs,
        )
    return calc


def sized_space_from_scf(
    mf: SCFclass,
    size: Union[int, tuple[int, int]],
    state: Optional[Union[int, tuple[int, int]]] = None,
    sort_mos: bool = False,
    mp2_kwargs: Optional[dict[str, Any]] = None,
    switch_dmrg: int = 12,
    dmrg_kwargs: Optional[dict[str, Any]] = None,
    fci_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = True,
    ci_nroots: int = 1,
) -> ActiveSpace:
    """Select an active space of certain size starting from an SCF calculation.

    The function executes the following steps:
        1)  Calculate MP2 natural orbitals using the SCF object provided in the input.
        2)  Select an initial set of MP2 natural orbitals based on their eigenvalues.
        3)  Perform a DMRGCI or CASCI calculation (depending on the number of active orbitals in
            the initial space) using the previously determined orbital subset.
        4)  Select the final active space based on the pair information.

    The output of the function is an ActiveSpace instance that contains:
        - the number of active electrons,
        - the list of MP2 natural orbitals selected for the active space,
        - and the MO coefficient matrix containing all (active and inactive) MP2 natural orbitals.

    The columns representing the inactive orbitals in the MO coefficient matrix are ordered
    according to the convention of PySCF: occupied MOs to the left and virtual MOs to the right.
    By default, the original ordering of the MP2 natural orbitals is preserved. This can be
    changed by setting reorder_mos to True: then, the columns of the MO coefficients are reordered
    such that they can be passed directly to CASCI/CASSCF without any further sorting.

    This function does not provide active spaces suggestions for state-averaged calculations, but
    for individual (excited) electronic states only. By default an active space is suggested for
    the ground state in the same spin configuration as the given SCF calculation. If a value is
    provided for 'state', the function performs an active space selection for the specific spin and
    electronic state.

    Args:
        mf:                 RHF or UHF object containing converged orbitals.
        size:               Requested number of active orbitals M, or number of active electrons N
                            and number of active orbitals as a CAS tuple (N, M).
        state:              Electronic state to construct the active space for.
                            None (default): Ground state for the same spin as in mf.
                            Integer n: electronic state n (indexed from 0), same spin as in mf.
                            Tuple (s, n): electronic state n (indexed from 0) for spin s.
        sort_mos:           If True, reorder the MO columns as occupied, active, virtual.
        mp2_kwargs:         Keyword arguments for MP2 orbital selection (see
                            'MP2NatorbPreselection').
        switch_dmrg:        Number of orbitals to switch between the regular FCI solver and DMRG.
        dmrg_kwargs:        Further keyword arguments to be passed to the DMRG fcisolver.
        fci_kwargs:         Further keyword arguments to be passed to CASCI.fcisolver.
        verbose:            Print information if set to True.
        ci_nroots:          Number of CI roots to use in CI/DMRG calculation. When several roots
                            are requested the CI solver may occasionally converge on the wrong
                            eigenvector. Use this option to solve for more roots to improve
                            convergence. This option has no direct consequence for the active space
                            selection.

    Returns:
        Active space including MO indices of active orbitals and MO coefficients
    """
    if mp2_kwargs is None:
        mp2_kwargs = {}
    mol = mf.mol

    if state is None:
        spin, root = (mol.spin, 0)
    elif isinstance(state, int):
        spin, root = (mol.spin, state)
    else:
        spin, root = state

    target_norb = target_nel = None
    if isinstance(size, int):
        target_norb = size
    elif isinstance(size, tuple):
        target_nel, target_norb = size

    # Select an initial set of the resulting MOs for the subsequent calculation.
    loghead("Calculating MP2 natural orbitals", verbose)
    preselection = MP2NatorbPreselection(scf=mf, **mp2_kwargs)
    space_mp2 = preselection.select()
    loginfo("", verbose)
    loginfo(
        (
            f"-> Selected initial orbital window of {space_mp2.nel:d} electrons in "
            f"{space_mp2.norb:d} MP2 natural orbitals."
        ),
        verbose,
    )

    # Determine active spaces for a specific spin state and electronic state.
    nroots = max(root + 1, ci_nroots)
    loghead("Running calculation", verbose)
    loginfo(f"spin = {spin:d}, requested root = {root:d}", verbose)
    loginfo(f"total number of roots calculated = {nroots:d}", verbose)
    loginfo("", verbose)

    calc = create_asf_switched_cisolver(
        mol=mol,
        initial_space=space_mp2,
        spin=spin,
        nroots=nroots,
        switch_dmrg=switch_dmrg,
        dmrg_kwargs=dmrg_kwargs,
        fci_kwargs=fci_kwargs,
    )
    calc.calculate()
    loginfo("", verbose)
    loginfo(f"Orbital selection for state {root:d}:", verbose)
    loginfo("", verbose)

    space = calc.find_one_sized(root=root, norb=target_norb, nel=target_nel)

    print_mo_table(
        calc.one_orbital_density(root=root),
        mo_list=list(calc.mo_list),
        selections={"a": space.mo_list},
    )

    loghead("Active space selection finished", verbose)
    loginfo(
        f"-> Selected an active space of {space.nel:d} electrons in {space.norb:d} orbitals.",
        verbose,
    )

    if sort_mos:
        act_sorted, mo_sorted = reorder_mos(mol, space.nel, space.mo_list, space_mp2.mo_coeff)
        return ActiveSpace(nel=space.nel, mo_list=act_sorted, mo_coeff=mo_sorted)

    return space


def sized_space_from_mol(
    mol: Mole,
    size: Union[int, tuple[int, int]],
    max_scf_restarts: int = DEFAULT_MAX_RESTARTS,
    scf_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = True,
    **kwargs,
) -> ActiveSpace:
    """Select an active space of certain size starting from a molecule.

    This function attempts to calculate a stable UHF solution for the Mole object provided.
    Subsequently, sized_space_from_scf(...) is called to calculate MP2 natural orbitals, and to
    select an active space from those.

    For more details, please refer to sized_space_from_scf(...).

    Args:
        mol:                Mole instance containing the molecular data.
        size:               Requested number of active orbitals M, or number of active electrons N
                            and number of active orbitals as a CAS tuple (N, M).
        max_scf_restarts:   Maximal number of times to restart when an unconverged or unstable
                            solution is found.
        scf_kwargs:         Dictionary of options to be set for the SCF object.
                            Note: this function deviates from PySCF defaults for a few settings.
        stability_analysis: Perform a stability analysis of the SCF solution if True.
                            Re-start the SCF calculation once of the solution is unstable.
        verbose:            Print information if set to True.
        **kwargs:           Keyword arguments to be passed to sized_space_from_scf(...).

    Returns:
        Active space suggestion as provided by sized_space_from_scf(...).

    Raises:
        SCFError: SCF calculation failed to converge or is unstable.
    """
    # Create a copy of the dictionary with the SCF settings. Include any entries from scf_kwargs.
    # Values provided in scf_kwargs always take priority over the defaults.
    scf_settings = DEFAULT_SCF_SETTINGS.copy()
    if scf_kwargs is not None:
        scf_settings.update(scf_kwargs)

    # Perform the UHF calculation.
    loghead("Calculating UHF orbitals", verbose)
    mf = stable_scf(
        mol,
        with_uhf=True,
        max_restarts=max_scf_restarts,
        scf_kwargs=scf_kwargs,
        basic_print=verbose,
    )

    return sized_space_from_scf(mf, size=size, verbose=verbose, **kwargs)


def find_from_scf(
    mf: SCFclass,
    max_norb: Optional[int] = None,
    min_norb: Optional[int] = None,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    states: Optional[Union[int, list[tuple[int, int]]]] = None,
    sort_mos: bool = False,
    mp2_kwargs: Optional[dict[str, Any]] = None,
    switch_dmrg: int = 12,
    dmrg_kwargs: Optional[dict[str, Any]] = None,
    fci_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = True,
    ci_nroots: int = 1,
) -> ActiveSpace:
    """Select an active space with an entropy threshold, starting from an SCF calculation.

    The function executes the following steps:
        1)  Calculate MP2 natural orbitals using the SCF object provided in the input.
        2)  Select an initial set of MP2 natural orbitals based on their eigenvalues.
        3)  Perform a DMRGCI or CASCI calculation (depending on the number of active orbitals in
            the initial space) using the previously determined orbital subset.
        4)  Among all sensible active spaces, select a solution such that the entropies of all
            active orbitals are above the entropy threshold, and simultaneously the pair
            information sum is maximized.

    The output of the function is an ActiveSpace instance that contains:
        - the number of active electrons,
        - the list of MP2 natural orbitals selected for the active space,
        - and the MO coefficient matrix containing all (active and inactive) MP2 natural orbitals.

    The columns representing the inactive orbitals in the MO coefficient matrix are ordered
    according to the convention of PySCF: occupied MOs to the left and virtual MOs to the right.
    By default, the original ordering of the MP2 natural orbitals is preserved. This can be
    changed by setting sort_mos to True: then, the columns of the MO coefficients are reordered
    such that they can be passed directly to CASCI/CASSCF without any further sorting.

    This function does not provide active spaces for individual excited electronic states.
    However, it can be used to obtain active space suggestions for state-averaged calculations.
    If a value is provided for 'states', the function performs active space selections for all spin
    and electronic states, and returns a representation of the combined active space.

    Args:
        mf:                 RHF or UHF object containing converged orbitals.
        max_norb:           Maximum number of active orbitals per root.
        min_norb:           Minimum number of active orbitals per root.
        entropy_threshold:  Entropy threshold to select a single active space. Among all reasonable
                            active spaces, a space with the lowest entropy above this threshold and
                            with maximal pair information sum is selected as the final choice.
        states:             Electronic states to construct the active space for.
                            None (default): Ground state for the same spin as in mf.
                            Integer n: lowest n electronic states, same spin as in mf.
                            List of Tuples [(s, n)]: lowest n electronic states for each spin s.
        sort_mos:           If True, reorder the MO columns as occupied, active, virtual.
        mp2_kwargs:         Keyword arguments for MP2 orbital selection (see
                            'MP2NatorbPreselection').
        switch_dmrg:        Number of orbitals to switch between the regular FCI solver and DMRG.
        dmrg_kwargs:        Further keyword arguments to be passed to the DMRG fcisolver.
        fci_kwargs:         Further keyword arguments to be passed to CASCI.fcisolver.
        verbose:            Print information if set to True.
        ci_nroots:          Number of CI roots to use in CI/DMRG calculation. When several roots
                            are requested the CI solver may occasionally converge on the wrong
                            eigenvector. Use this option to solve for more roots to improve
                            convergence. This option has no direct consequence for the active space
                            selection.

    Returns:
        Active space including MO indices of active orbitals and MO coefficients
    """
    # Avoiding trouble with mutable default arguments...
    if mp2_kwargs is None:
        mp2_kwargs = {}
    mol = mf.mol

    if states is None:
        spin_roots = [(mol.spin, 1)]
    elif isinstance(states, int):
        spin_roots = [(mol.spin, states)]
    else:
        spin_roots = states

    space_filters: list[FilterFunction] = []
    if min_norb is not None:

        def truncate_min_norb(space: MOListInfo) -> bool:
            return len(space.mo_list) >= min_norb

        space_filters.append(truncate_min_norb)
    if max_norb is not None:

        def truncate_max_norb(space: MOListInfo) -> bool:
            return len(space.mo_list) <= max_norb

        space_filters.append(truncate_max_norb)

    # Select an initial set of the resulting MOs for the subsequent calculation.
    loghead("Calculating MP2 natural orbitals", verbose)
    preselection = MP2NatorbPreselection(scf=mf, **mp2_kwargs)
    space_mp2 = preselection.select()
    loginfo("", verbose)
    loginfo(
        (
            f"-> Selected initial orbital window of {space_mp2.nel:d} electrons in "
            f"{space_mp2.norb:d} MP2 natural orbitals."
        ),
        verbose,
    )

    # Determine active spaces for all spin states and electronic states.
    nel = 0
    mo_list: list[int] = []
    for spin, nroots in spin_roots:
        nroots_total = max(nroots, ci_nroots)
        loghead("Running calculation", verbose)
        loginfo(f"spin = {spin:d}, number of roots = {nroots:d}", verbose)
        loginfo(f"total number of roots calculated = {nroots_total:d}", verbose)
        loginfo("", verbose)

        calc = create_asf_switched_cisolver(
            mol=mol,
            initial_space=space_mp2,
            spin=spin,
            nroots=nroots_total,
            switch_dmrg=switch_dmrg,
            dmrg_kwargs=dmrg_kwargs,
            fci_kwargs=fci_kwargs,
        )
        calc.calculate()

        for root in range(nroots):
            loginfo("", verbose)
            loginfo(f"Orbital selection for state {root:d}:", verbose)
            loginfo("", verbose)

            space = calc.find_one_entropy(
                root=root,
                entropy_threshold=entropy_threshold,
                filters=space_filters,
            )

            print_mo_table(
                calc.one_orbital_density(root=root),
                mo_list=list(calc.mo_list),
                selections={"a": space.mo_list},
            )

            # Merge current active space with the already existing space.
            nel, mo_list = merge_active_spaces(mol, nel, mo_list, space.nel, space.mo_list)

    loghead("Active space selection finished", verbose)
    loginfo(
        f"-> Selected an active space of {nel:d} electrons in {len(mo_list):d} orbitals.", verbose
    )

    if sort_mos:
        act_sorted, mo_sorted = reorder_mos(mol, nel, mo_list, space_mp2.mo_coeff)
        return ActiveSpace(nel=nel, mo_list=act_sorted, mo_coeff=mo_sorted)

    return ActiveSpace(nel=nel, mo_list=mo_list, mo_coeff=space_mp2.mo_coeff)


def find_from_mol(
    mol: Mole,
    max_norb: Optional[int] = None,
    min_norb: Optional[int] = None,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    max_scf_restarts: int = DEFAULT_MAX_RESTARTS,
    scf_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = True,
    **kwargs,
) -> ActiveSpace:
    """Select an active space with an entropy threshold, starting from a Molecule.

    This function attempts to calculate a stable UHF solution for the Mole object provided.
    Subsequently, find_from_scf(...) is called to calculate MP2 natural orbitals, and to
    select an active space from those.

    For more details, please refer to find_from_scf(...).

    Args:
        mol:                Mole instance containing the molecular data.
        max_norb:           Maximum number of active orbitals per root.
        min_norb:           Minimum number of active orbitals per root.
        entropy_threshold:  Entropy threshold to select a single active space. Among all reasonable
                            active spaces, a space with the lowest entropy above this threshold and
                            with maximal pair information sum is selected as the final choice.
        max_scf_restarts:   Maximal number of times to restart when an unconverged or unstable
                            solution is found.
        scf_kwargs:         Dictionary of options to be set for the SCF object.
                            Note: this function deviates from PySCF defaults for a few settings.
        stability_analysis: Perform a stability analysis of the SCF solution if True.
                            Re-start the SCF calculation once of the solution is unstable.
        verbose:            Print information if set to True.
        **kwargs:           Keyword arguments to be passed to find_from_scf(...).

    Returns:
        Active space suggestion as provided by find_from_scf(...).

    Raises:
        SCFError: SCF calculation failed to converge or is unstable.
    """
    # Create a copy of the dictionary with the SCF settings. Include any entries from scf_kwargs.
    # Values provided in scf_kwargs always take priority over the defaults.
    scf_settings = DEFAULT_SCF_SETTINGS.copy()
    if scf_kwargs is not None:
        scf_settings.update(scf_kwargs)

    # Perform the UHF calculation.
    loghead("Calculating UHF orbitals", verbose)
    mf = stable_scf(
        mol,
        with_uhf=True,
        max_restarts=max_scf_restarts,
        scf_kwargs=scf_kwargs,
        basic_print=verbose,
    )

    return find_from_scf(
        mf,
        entropy_threshold=entropy_threshold,
        min_norb=min_norb,
        max_norb=max_norb,
        verbose=verbose,
        **kwargs,
    )
