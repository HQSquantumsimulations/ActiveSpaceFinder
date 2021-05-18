from typing import Union

import numpy as np

from tempfile import TemporaryDirectory

from pyscf.mcscf import CASCI
from pyscf.dmrgscf import DMRGCI
from pyscf.lib import num_threads
from pyscf.lib.parameters import MAX_MEMORY, TMPDIR
from pyscf.gto import Mole

from asf.casci import ASFCI


class ASFDMRG(ASFCI):
    """
    Sets up and performs DMRG-CASCI calculations for entropy-based active space selection.
    """

    def __init__(self,
                 mol: Mole,
                 mo_coeff: np.ndarray,
                 nel: int = None,
                 norb: int = None) -> None:
        """
        Sets initial parameters for a DMRG-CASCI calculation with a specific number of
        electrons and orbitals. Without giving a number of electrons and orbitals,
        it initializes a DMRG-full-CI calculation.

        Args:
            mol: molecular information
            mo_coeff: spin-restricted molecular orbital coefficients
            nel: number of electrons in the initial orbital space
            norb: number of orbitals in the initial space
        """
        super().__init__(mol, mo_coeff, nel, norb)
        self.tmpdir: Union[TemporaryDirectory, None] = None

    def runCI(self, *args, **kwargs):
        """
        Disabled with DMRG.
        """
        raise NotImplementedError('The runDMRG method should be used.')

    def runDMRG(self,
                maxM: int = 500,
                tol: float = 1.0e-6,
                **kwargs) -> None:
        """
        Run the DMRG-(CAS)CI calculation for subsequent orbital selection.

        Args:
            maxM: bond dimension
            tol: tolerance
            **kwargs: all other arguments used in the DMRGCI class
        """
        casci = CASCI(self.mol, self.norb, self.nel)

        # PySCF's interface to Block asks for some trickery.
        num_thrds = num_threads()
        memory = MAX_MEMORY * 0.001
        casci.fcisolver = \
            DMRGCI(mol=self.mol, maxM=maxM, tol=tol, num_thrds=num_thrds, memory=memory)
        casci.fcisolver.spin = self.mol.spin

        for key, value in kwargs.items():
            setattr(casci.fcisolver, key, value)

        # Block demands a scratch directory on disk. For this purpose, we create
        # a temporary directory and tie its lifetime to that of the ASFDMRG instance.
        # It is removed automatically once the ASFDMRG instance is garbage collected.
        if 'scratchDirectory' not in kwargs:
            tmpdir = TemporaryDirectory(dir=TMPDIR)
            casci.fcisolver.scratchDirectory = tmpdir.name
            self.tmpdir = tmpdir

        # Run the actual DMRG calculation.
        if self.mo_list is None:
            casci_mos = self.mo_coeff
            self.mo_list = list(range(casci.ncore, casci.ncore + casci.ncas))
        else:
            casci_mos = casci.sort_mo(caslst=self.mo_list, mo_coeff=self.mo_coeff, base=0)
        casci.kernel(mo_coeff=casci_mos)

        self.casci = casci

    def getOrbDens(self, root: int = 0) -> np.ndarray:
        """
        Determines the one-orbital density for the previously calculated DMRG-(CAS)CI wave function.

        Args:
            root: state that the density is calculated for

        Returns:
            The one-orbital density for all orbitals as an N x 4 numpy array.
            Order of the columns is empty, spin-up, spin-down, doubly occupied.
        """
        assert(self.casci is not None)

        ncore = self.casci.ncore
        ncas = self.casci.ncas
        nmo = self.casci.mo_coeff.shape[1]

        # We need to obtain the RDMs somewhat differently from normal CASCI because of the way
        # that the interface of PySCF to Block is currently written.
        # alpha and beta 1-RDMs in the active space
        rdm1a, rdm1b = DMRGCI.make_rdm1s(self.casci.fcisolver, root, self.norb, self.nel)
        # spin-free 2-RDM in the active space
        _, rdm2 = DMRGCI.make_rdm12(self.casci.fcisolver, root, self.norb, self.nel)

        assert(rdm1a.shape == (ncas, ncas))
        assert(rdm1b.shape == (ncas, ncas))
        assert(rdm2.shape == (ncas, ncas, ncas, ncas))

        orbdens = np.zeros((nmo, 4))

        # for the sake of completeness, set one-orbital densities outside the active space.
        orbdens[:ncore, 3] = 1.0
        orbdens[(ncore + ncas):, 0] = 1.0

        # the relevant part: one-orbital density inside the active space
        for i in range(ncas):
            rdm2s = 0.5 * rdm2[i, i, i, i]
            orbdens[ncore + i, 0] = 1.0 - rdm1a[i, i] - rdm1b[i, i] + rdm2s
            orbdens[ncore + i, 1] = rdm1a[i, i] - rdm2s
            orbdens[ncore + i, 2] = rdm1b[i, i] - rdm2s
            orbdens[ncore + i, 3] = rdm2s

        return orbdens
