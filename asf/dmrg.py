# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Active space finder functionality based on DMRG."""

from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from pyscf.dmrgscf import DMRGCI
from pyscf.gto import Mole
from pyscf.lib import num_threads
from pyscf.lib.parameters import MAX_MEMORY, TMPDIR
from pyscf.mcscf import CASCI

from .asfbase import ASFBase, inactive_orbital_lists
from .pairinfo import cumulant2b
from .utility import orbdens_from_rdm, rdm1s_from_rdm12


class ASFDMRG(ASFBase):
    """Sets up and performs DMRG-CASCI calculations for entropy-based active space selection."""

    def __init__(
        self,
        mol: Mole,
        mo_coeff: np.ndarray,
        *args,
        nroots: int = 1,
        maxM: int = 500,
        tol: float = 1.0e-6,
        fcisolver_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Initializes the object. Refer to the ASFBase class for details.

        Args:
            mol:              Molecule object
            mo_coeff:         Spin-restricted molecular orbital coefficients
                              (AOs in rows, MOs in columns)
            nroots:           number of electronic states to calculate
            maxM:             bond dimension
            tol:              tolerance
            fcisolver_kwargs: all other arguments used in the DMRGCI class (see `pyscf.dmrgscf`);
                arguments from this dictionary take precedence if provided
            *args:            positional arguments
            **kwargs:         keyword arguments
        """
        super().__init__(mol, mo_coeff, *args, **kwargs)
        self.casci: Optional[CASCI] = None
        self.tmpdir: Optional[TemporaryDirectory] = None
        self.fcisolver_kwargs = {"nroots": nroots, "maxM": maxM, "tol": tol}
        if fcisolver_kwargs is not None:
            self.fcisolver_kwargs.update(fcisolver_kwargs)

    def calculate(self) -> None:
        """Run the DMRG-(CAS)CI calculation for subsequent orbital selection.

        Raises:
            ValueError: bad arguments
        """
        casci = CASCI(self.mol, self.norb, self.nel)

        # PySCF's interface to Block asks for some trickery.
        num_thrds = num_threads()
        memory = MAX_MEMORY * 0.001
        casci.fcisolver = DMRGCI(
            mol=self.mol,
            maxM=self.fcisolver_kwargs["maxM"],
            tol=self.fcisolver_kwargs["tol"],
            num_thrds=num_thrds,
            memory=memory,
        )
        casci.fcisolver.nroots = self.fcisolver_kwargs["nroots"]

        # Setting an arbitrary spin does not work due to PySCF bug.
        # Correct spin from self.mol is used, anyway.

        for key, value in self.fcisolver_kwargs.items():
            if key in ["maxM", "tol", "nroots"]:
                continue
            setattr(casci.fcisolver, key, value)

        # Block demands a scratch directory on disk. For this purpose, we create
        # a temporary directory and tie its lifetime to that of the ASFDMRG instance.
        # It is removed automatically once the ASFDMRG instance is garbage collected.
        if "scratchDirectory" not in self.fcisolver_kwargs:
            if self.fcisolver_kwargs.get("restart", False):
                # If the restart feature is used, a temporary directory needs to exist on disk.
                if self.tmpdir is None:
                    raise ValueError("Requested restart feature without previous DMRG results.")
            else:
                # In a normal run, create the temporary directory.
                self.tmpdir = TemporaryDirectory(dir=TMPDIR)
            casci.fcisolver.scratchDirectory = self.tmpdir.name

        # Run the actual DMRG calculation.
        mo_sorted = casci.sort_mo(caslst=self.mo_list, mo_coeff=self.mo_coeff, base=0)
        casci.kernel(mo_coeff=mo_sorted)

        self.casci = casci

    def one_orbital_density(self, root: int = 0) -> np.ndarray:
        """Determines the one-orbital density for the calculated DMRG-(CAS)CI wave function.

        Args:
            root: state that the density is calculated for

        Returns:
            The one-orbital density for all orbitals as an N x 4 numpy array.
            Order of the columns is empty, spin-up, spin-down, doubly occupied.

        Raises:
            Exception: various errors
        """
        if self.casci is None:
            raise Exception("DMRG calculation must be performed first to calculate the density.")
        if root < 0 or root >= self.casci.fcisolver.nroots:
            raise Exception("Invalid root requested.")

        ncore = self.casci.ncore
        ncas = self.casci.ncas
        nmo = self.casci.mo_coeff.shape[1]

        # Only spin-free RDMs are provided by Block. Get the 1-RDM and 2-RDM.
        rdm1, rdm2 = DMRGCI.make_rdm12(self.casci.fcisolver, root, self.norb, self.nel)
        # alpha and beta 1-RDMs in the active space
        # workaround for the PySCF interface bug
        rdm1a, rdm1b = self._calc_rdm1s(root, (rdm1, rdm2))

        if not (rdm1a.shape == (ncas, ncas)):
            raise Exception("bad dimensions")
        if not (rdm1b.shape == (ncas, ncas)):
            raise Exception("bad dimensions")
        if not (rdm2.shape == (ncas, ncas, ncas, ncas)):
            raise Exception("bad dimensions")

        # PySCF orders orbitals as core, active, virtual. -> Map them back onto the original
        # ordering.
        act_list = self.mo_list
        core_list, virt_list = inactive_orbital_lists(ncore, nmo, act_list)

        # one-orbital density
        orbdens = np.zeros((nmo, 4))
        # the relevant part: one-orbital density inside the active space
        orbdens[act_list, :] = orbdens_from_rdm(rdm1a, rdm1b, 0.5 * rdm2)
        # for the sake of completeness, set one-orbital densities outside the active space.
        orbdens[core_list, 3] = 1.0
        orbdens[virt_list, 0] = 1.0

        return orbdens

    def cumulant_4idx(self, root: int = 0) -> np.ndarray:
        """Calculates the complete two-body cumulant in the active space.

        Args:
            root: the root for which the cumulant is computed

        Returns:
            Four-index tensor with the cumulant.
            Note: 0 labels the first active orbital.

        Raises:
            Exception: input error
        """
        if self.casci is None:
            raise Exception("Object has got no CASCI attribute.")
        rdm1, rdm2 = DMRGCI.make_rdm12(self.casci.fcisolver, root, self.norb, self.nel)
        rdm1a, rdm1b = self._calc_rdm1s(root, (rdm1, rdm2))
        return cumulant2b(rdm1a, rdm1b, rdm2)

    def diagonal_cumulant(self, root: int = 0, full_space: bool = True) -> np.ndarray:
        """The 'diagonal' elements of the two-body cumulant, which derive from <p^+ q^+ q p>.

        Optionally, the matrix is filled up with zeros to cover the entire set of orbitals.

        Args:
            root: the root for which the density is computed
            full_space: if true, expand the cumulant to cover the full MO space
                        (filled up with zeros)

        Returns:
            Matrix with the relevant entries of the cumulant.
            Note: 0 labels the first active orbital.

        Raises:
            Exception: input error
        """
        if self.casci is None:
            raise Exception("Object has got no CASCI attribute.")
        ncas = self.casci.ncas
        nmo = self.casci.mo_coeff.shape[1]
        norb = self.norb
        act_list = self.mo_list

        cum4idx = self.cumulant_4idx(root=root)
        if full_space:
            cumdiag = np.zeros((nmo, nmo))
            for p in range(ncas):
                for q in range(ncas):
                    cumdiag[act_list[p], act_list[q]] = cum4idx[p, p, q, q]
        else:
            cumdiag = np.zeros((norb, norb))
            for p in range(norb):
                for q in range(norb):
                    cumdiag[p, q] = cum4idx[p, p, q, q]
        return cumdiag

    def _calc_rdm1s(
        self, root: int = 0, rdm12: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the spin components of the 1-RDM in the active space.

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
            raise ValueError("Object has no attribute with DMRG-CASCI results.")
        if rdm12 is None:
            rdm1, rdm2 = DMRGCI.make_rdm12(self.casci.fcisolver, root, self.norb, self.nel)
        else:
            rdm1, rdm2 = rdm12
        # fcisolver.spin is not set properly in the Block interface
        S = self.mol.spin * 0.5
        return rdm1s_from_rdm12(self.nel, S, rdm1, rdm2)

    def rdm1s(self, root: int = 0) -> np.ndarray:
        """Returns the active one-electron spin density matrix as an 2 x N x N array."""
        return np.array(self._calc_rdm1s(root=root))

    def rdm2(self, root: int = 0) -> np.ndarray:
        """Returns the active two-electron spin-free density matrix as a four-dimensional array."""
        if not self.casci:
            raise ValueError("Object has no attribute with DMRG-CASCI results.")
        _, rdm2 = DMRGCI.make_rdm12(self.casci.fcisolver, root, self.norb, self.nel)
        return np.array(rdm2)
