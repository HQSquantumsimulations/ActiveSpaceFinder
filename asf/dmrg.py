# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Active space finder functionality based on DMRG."""

from typing import Union, Tuple
from tempfile import TemporaryDirectory
import numpy as np

from pyscf.mcscf import CASCI
from pyscf.dmrgscf import DMRGCI
from pyscf.lib import num_threads
from pyscf.lib.parameters import MAX_MEMORY, TMPDIR

from .asf import ASFBase, inactive_orbital_lists
from asf.utility import rdm1s_from_rdm12


class ASFDMRG(ASFBase):
    """Sets up and performs DMRG-CASCI calculations for entropy-based active space selection."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the object. Refer to the ASFBase class for details.

        Args:
            *args: positional arguments
            **kwargs: keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.casci: Union[CASCI, None] = None
        self.tmpdir: Union[TemporaryDirectory, None] = None

    def runDMRG(self,
                nroots: int = 1,
                maxM: int = 500,
                tol: float = 1.0e-6,
                **kwargs) -> None:
        """
        Run the DMRG-(CAS)CI calculation for subsequent orbital selection.

        Args:
            nroots: number of electronic state to calculate
            maxM: bond dimension
            tol: tolerance
            **kwargs: all other arguments used in the DMRGCI class

        Raises:
            ValueError: bad arguments
        """
        casci = CASCI(self.mol, self.norb, self.nel)

        # PySCF's interface to Block asks for some trickery.
        num_thrds = num_threads()
        memory = MAX_MEMORY * 0.001
        casci.fcisolver = \
            DMRGCI(mol=self.mol, maxM=maxM, tol=tol, num_thrds=num_thrds, memory=memory)
        casci.fcisolver.nroots = nroots

        # Setting an arbitrary spin does not work due to PySCF bug.
        # Correct spin from self.mol is used, anyway.
        # casci.fcisolver.spin = self.mol.spin

        for key, value in kwargs.items():
            setattr(casci.fcisolver, key, value)

        # Block demands a scratch directory on disk. For this purpose, we create
        # a temporary directory and tie its lifetime to that of the ASFDMRG instance.
        # It is removed automatically once the ASFDMRG instance is garbage collected.
        if 'scratchDirectory' not in kwargs:
            if kwargs.get('restart', False):
                # If the restart feature is used, a temporary directory needs to exist on disk.
                if self.tmpdir is None:
                    raise ValueError('Requested restart feature without previous DMRG results.')
            else:
                # In a normal run, create the temporary directory.
                self.tmpdir = TemporaryDirectory(dir=TMPDIR)
            casci.fcisolver.scratchDirectory = self.tmpdir.name

        # Run the actual DMRG calculation.
        mo_sorted = casci.sort_mo(caslst=self.mo_list, mo_coeff=self.mo_coeff, base=0)
        casci.kernel(mo_coeff=mo_sorted)

        self.casci = casci

    def calculate(self, *args, **kwargs) -> None:
        """Wrapper for the runDMRG method.

        Args:
            *args: positional arguments for self.runDMRG(...)
            **kwargs: keyword arguments for self.runDMRG(...)
        """
        self.runDMRG(*args, **kwargs)

    def getOrbDens(self, root: int = 0) -> np.ndarray:
        """
        Determines the one-orbital density for the previously calculated DMRG-(CAS)CI wave function.

        Args:
            root: state that the density is calculated for

        Returns:
            The one-orbital density for all orbitals as an N x 4 numpy array.
            Order of the columns is empty, spin-up, spin-down, doubly occupied.

        Raises:
            Exception: various errors
        """
        if self.casci is None:
            raise Exception('DMRG calculation must be performed first to calculate the density.')
        if root < 0 or root >= self.casci.fcisolver.nroots:
            raise Exception('Invalid root requested.')

        ncore = self.casci.ncore
        ncas = self.casci.ncas
        nmo = self.casci.mo_coeff.shape[1]

        # Only spin-free RDMs are provided by Block. Get the 1-RDM and 2-RDM.
        rdm1, rdm2 = DMRGCI.make_rdm12(self.casci.fcisolver, root, self.norb, self.nel)
        # alpha and beta 1-RDMs in the active space
        # workaround for the PySCF interface bug
        rdm1a, rdm1b = self._calc_rdm1s(root, (rdm1, rdm2))

        if not (rdm1a.shape == (ncas, ncas)):
            raise Exception('bad dimensions')
        if not (rdm1b.shape == (ncas, ncas)):
            raise Exception('bad dimensions')
        if not (rdm2.shape == (ncas, ncas, ncas, ncas)):
            raise Exception('bad dimensions')

        # PySCF orders orbitals as core, active, virtual. -> Map them back onto the original
        # ordering.
        act_list = self.mo_list
        core_list, virt_list = inactive_orbital_lists(ncore, nmo, act_list)

        # one-orbital density
        orbdens = np.zeros((nmo, 4))
        # for the sake of completeness, set one-orbital densities outside the active space.
        orbdens[core_list, 3] = 1.0
        orbdens[virt_list, 0] = 1.0

        # the relevant part: one-orbital density inside the active space
        for i in range(ncas):
            rdm2s = 0.5 * rdm2[i, i, i, i]
            orbdens[act_list[i], 0] = 1.0 - rdm1a[i, i] - rdm1b[i, i] + rdm2s
            orbdens[act_list[i], 1] = rdm1a[i, i] - rdm2s
            orbdens[act_list[i], 2] = rdm1b[i, i] - rdm2s
            orbdens[act_list[i], 3] = rdm2s

        return orbdens

    def _calc_rdm1s(self,
                    root: int = 0,
                    rdm12: Tuple[np.ndarray, np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the spin components of the 1-RDM in the active space.

        Workaround for a bug in the PySCF interface to Block.

        Args:
            root:   state to calculate the density for
            rdm12:  (optionally), tuple containing the 1-RDM and 2-RDM

        Returns:
            alpha-1-RDM, beta-1-RDM

        Raises:
            ValueError: bad input
        """
        if self.casci is None:
            raise ValueError('DMRG-CI object not present.')
        if rdm12 is None:
            rdm1, rdm2 = DMRGCI.make_rdm12(self.casci.fcisolver, root, self.norb, self.nel)
        else:
            rdm1, rdm2 = rdm12
        # fcisolver.spin is not set properly in the Block interface
        S = self.mol.spin * 0.5
        return rdm1s_from_rdm12(self.nel, S, rdm1, rdm2)
