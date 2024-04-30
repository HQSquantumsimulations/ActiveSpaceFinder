# Copyright © 2023 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Definitions and functions centered around the Self-Consistent Field Method."""


from enum import Enum
from typing import Any, Optional, Union

from pyscf.gto import Mole
from pyscf.scf import RHF, UHF
from pyscf.scf.hf import RHF as RHFClass
from pyscf.scf.uhf import UHF as UHFClass

from .utility import loginfo

DEFAULT_SCF_SETTINGS = {"max_cycle": 100, "conv_tol": 1.0e-7, "level_shift": 0.2, "damp": 0.2}
"""Default settings for PySCF's mean-field classes (e.g. RHF).
"""

# Be orderly about the SCF types we can encounter.
SCFtype = Enum("SCFtype", "RHF UHF ROHF")


class SCFError(Exception):
    """Failed to obtain a converged and stable SCF solution."""


DEFAULT_MAX_RESTARTS = 5
"""Number of times to restart an SCF calculation if the previous attempt failed.
"""


def stable_scf(
    mol: Mole,
    with_uhf: bool = True,
    newton_backup: bool = True,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    scf_kwargs: Optional[dict[str, Any]] = None,
    basic_print: bool = True,
) -> Union[RHFClass, UHFClass]:
    """Perform Hartree-Fock calculations with stability analysis on a Mole object.

    If an SCF calculation did not converge or if the solution is internally unstable, the SCF
    calculation is restarted.

    Many settings of the SCF object can be controlled via the scf_kwargs dictionary.

    Example 1: PySCF does not print output for individual SCF iterations with by default. The level
    of output can be controlled either via the Mole object, or via the 'verbose' attribute of an
    SCF object:

        scf_kwargs = {"verbose": pyscf.lig.logger.INFO}

    Example 2: the number of SCF iterations can be changed from its default value.

        scf_kwargs = {"max_cycle": 100}

    Example 3: the guess type for the SCF calculation can be modified, for example by reading the
    starting orbitals from a checkpoint file.

    IMPORTANT: if you use the keywords below, then the checkpoint file provided will be overwritten
    or deleted! Only do this with a disposable copy of the file.

        scf_kwargs = {
            "init_guess": "chkfile",
            "chkfile": "my_file.chk",
        }

    User documentation for SCF: https://pyscf.org/user/scf.html
    API documentation for RHF: https://pyscf.org/pyscf_api_docs/pyscf.scf.html#module-pyscf.scf.hf
    API documentation for UHF: https://pyscf.org/pyscf_api_docs/pyscf.scf.html#module-pyscf.scf.uhf

    Args:
        mol:            Instance of the PySCF Mole class.
        with_uhf:       Perform a UHF calculation if True, an RHF calculation if False.
        newton_backup:  Switch to Newton solver (more expensive) if a previous calculation failed
                        to converge if set to True. Set to False to disable.
        max_restarts:   Maximal number of times to restart after unconverged or unstable solutions.
        scf_kwargs:     Dictionary of attributes to set on the SCF object prior to performing the
                        calculation.
        basic_print:    Print information on steps initiated by this function if set to True.

    Returns:
        Either a UHF or an RHF object, depending on the input.

    Raises:
        SCFError:   No converged and stable result was achieved.
    """
    # Set up the SCF object.
    if with_uhf:
        mf = UHF(mol)
        mf.init_guess_breaksym = True
    else:
        if mol.spin != 0:
            raise ValueError("RHF calculation for a molecule with unpaired electrons requested.")
        mf = RHF(mol)

    if scf_kwargs:
        for name, value in scf_kwargs.items():
            setattr(mf, name, value)

    using_newton = False

    loginfo(f"-> Initiating a {'UHF' if with_uhf else 'RHF'} calculation.", basic_print)

    # Repeat SCF calculations until a converged and stable solution has been obtained.
    mf.kernel()
    for num_restart in range(max_restarts):

        mo_new, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)

        # If the calculation did not converge, switch to Newton unless that was disabled.
        if not mf.converged:
            loginfo("-> Calculation did not converge.", basic_print)
            if newton_backup and not using_newton:
                loginfo("-> Switching to Newton solver.", basic_print)
                mf = mf.newton()
                using_newton = True

        # If the calculation converged but the result is not stable, print info.
        elif not stable:
            loginfo("-> Restarting SCF due to instability of previous solution.", basic_print)

        # If the solution is both converged and stable, exit the loop.
        else:
            loginfo("-> The calculated SCF solution is converged and stable.", basic_print)
            break

        # Restart calculation with suggestion from stability analysis, if applicable.
        loginfo(f"-> Initiating SCF restart number {num_restart+1}.", basic_print)
        mf.kernel(dm0=mf.make_rdm1(mo_coeff=mo_new))
    else:
        raise SCFError(f"No stable/converged SCF solution found in {max_restarts} iterations.")

    return mf
