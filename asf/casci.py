# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Active space finder functionality based on regular CASCI."""

from typing import Union
import numpy as np
from pyscf.mcscf import CASCI
from .asf import ASFBase, inactive_orbital_lists


class ASFCI(ASFBase):
    """Sets up and performs CASCI calculations for entropy-based active space selection."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the object. Refer to the ASFBase for details.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.casci: Union[CASCI, None] = None

    def runCI(self,
              nroots: int = 1,
              spin_shift: float = None,
              **kwargs) -> None:
        """
        Run the (CAS-)CI calculation for subsequent orbital selection.

        Args:
            nroots: number of electronic states to calculate
            spin_shift: optionally, the shift for PySCF's fix_spin function
            **kwargs: arguments that can be passed on the the FCI solver
        """
        casci = CASCI(self.mol, self.norb, self.nel)
        if spin_shift is None:
            casci.fix_spin()
        else:
            casci.fix_spin(shift=spin_shift)
        casci.fcisolver.nroots = nroots

        for key, value in kwargs.items():
            setattr(casci.fcisolver, key, value)

        mo_sorted = casci.sort_mo(caslst=self.mo_list, mo_coeff=self.mo_coeff, base=0)
        casci.kernel(mo_coeff=mo_sorted)
        self.casci = casci

    def calculate(self, *args, **kwargs) -> None:
        """Wrapper for the runCI method.

        Args:
            *args: positional arguments for self.runCI(...)
            **kwargs: keyword arguments for self.runCI(...)
        """
        self.runCI(*args, **kwargs)

    def getOrbDens(self, root: int = 0) -> np.ndarray:
        """
        Determines the one-orbital density for the previously calculated (CAS-)CI wave function.

        Args:
            root: state that the density is calculated for

        Returns:
            The one-orbital density for all orbitals as an N x 4 numpy array.
            Order of the columns is empty, spin-up, spin-down, doubly occupied.

        Raises:
            Exception:  various errors
        """
        if self.casci is None:
            raise Exception('CASCI calculation must be performed first to calculate the density.')
        if root < 0 or root >= self.casci.fcisolver.nroots:
            raise Exception('Invalid root requested.')

        ncore = self.casci.ncore
        ncas = self.casci.ncas
        nel = self.casci.nelecas
        nmo = self.casci.mo_coeff.shape[1]

        # active space RDMs in spin-orbital basis
        fcisolver = self.casci.fcisolver
        if fcisolver.nstates == 1:
            civec = fcisolver.ci
        else:
            civec = fcisolver.ci[root]
        (rdm1a, rdm1b), (_, rdm2ab, _) = fcisolver.make_rdm12s(civec, ncas, nel)

        if not (rdm1a.shape == (ncas, ncas)):
            raise Exception('bad rdm1a dimensions')
        if not (rdm1b.shape == (ncas, ncas)):
            raise Exception('bad rdm1b dimensions')
        if not (rdm2ab.shape == (ncas, ncas, ncas, ncas)):
            raise Exception('bad rdm2ab dimensions')

        # PySCF orders orbitals as core, active, virtual. -> Map them back onto the original
        # ordering.
        act_list = self.mo_list
        core_list, virt_list = inactive_orbital_lists(ncore, nmo, act_list)

        # one-orbital density
        orbdens = np.zeros((nmo, 4))
        # for the sake of completeness, also set one-orbital densities outside the active space
        orbdens[core_list, 3] = 1.0
        orbdens[virt_list, 0] = 1.0

        # the relevant part: one-orbital density inside the active space
        for i in range(ncas):
            rdm2s = rdm2ab[i, i, i, i]
            orbdens[act_list[i], 0] = 1.0 - rdm1a[i, i] - rdm1b[i, i] + rdm2s
            orbdens[act_list[i], 1] = rdm1a[i, i] - rdm2s
            orbdens[act_list[i], 2] = rdm1b[i, i] - rdm2s
            orbdens[act_list[i], 3] = rdm2s

        return orbdens
