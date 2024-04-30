# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Active space finder functionality based on regular CASCI."""

from typing import Optional

import numpy as np
from pyscf.gto import Mole
from pyscf.mcscf import CASCI

from .asfbase import ASFBase, inactive_orbital_lists
from .pairinfo import cumulant2b
from .utility import orbdens_from_rdm


class ASFCI(ASFBase):
    """Sets up and performs CASCI calculations for entropy-based active space selection."""

    def __init__(
        self,
        mol: Mole,
        mo_coeff: np.ndarray,
        *args,
        nroots: int = 1,
        spin_shift: Optional[float] = None,
        fcisolver_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Initializes the object. Refer to the ASFBase for details.

        Args:
            mol:              Molecule object
            mo_coeff:         Spin-restricted molecular orbital coefficients
                              (AOs in rows, MOs in columns)
            nroots:           number of electronic states to calculate
            spin_shift:       optionally, the shift for PySCF's fix_spin function
            fcisolver_kwargs: arguments that can be passed to the FCI solver (see `pyscf.fci`);
                arguments from this dictionary take precedence if provided
            *args:            positional arguments
            **kwargs:         keyword arguments
        """
        super().__init__(mol, mo_coeff, *args, **kwargs)
        self.casci: Optional[CASCI] = None
        self.fcisolver_kwargs = {"nroots": nroots}
        if fcisolver_kwargs is not None:
            self.fcisolver_kwargs.update(fcisolver_kwargs)
        self.spin_shift = spin_shift

    def calculate(self) -> None:
        """Run the (CAS-)CI calculation for subsequent orbital selection."""
        casci = CASCI(self.mol, self.norb, self.nel)
        if self.spin_shift is None:
            casci.fix_spin()
        else:
            casci.fix_spin(shift=self.spin_shift)
        casci.fcisolver.nroots = self.fcisolver_kwargs["nroots"]

        for key, value in self.fcisolver_kwargs.items():
            if key in ["nroots"]:
                continue
            setattr(casci.fcisolver, key, value)

        mo_sorted = casci.sort_mo(caslst=self.mo_list, mo_coeff=self.mo_coeff, base=0)
        casci.kernel(mo_coeff=mo_sorted)
        self.casci = casci

    def one_orbital_density(self, root: int = 0) -> np.ndarray:
        """Determines the one-orbital density for the previously calculated (CAS-)CI wave function.

        Args:
            root: state that the density is calculated for

        Returns:
            The one-orbital density for all orbitals as an N x 4 numpy array.
            Order of the columns is empty, spin-up, spin-down, doubly occupied.

        Raises:
            Exception:  various errors
        """
        if self.casci is None:
            raise Exception("CASCI calculation must be performed first to calculate the density.")
        if root < 0 or root >= self.casci.fcisolver.nroots:
            raise Exception("Invalid root requested.")

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
            raise Exception("bad rdm1a dimensions")
        if not (rdm1b.shape == (ncas, ncas)):
            raise Exception("bad rdm1b dimensions")
        if not (rdm2ab.shape == (ncas, ncas, ncas, ncas)):
            raise Exception("bad rdm2ab dimensions")

        # PySCF orders orbitals as core, active, virtual. -> Map them back onto the original
        # ordering.
        act_list = self.mo_list
        core_list, virt_list = inactive_orbital_lists(ncore, nmo, act_list)

        # one-orbital density
        orbdens = np.zeros((nmo, 4))
        # the relevant part: one-orbital density inside the active space
        orbdens[act_list, :] = orbdens_from_rdm(rdm1a, rdm1b, rdm2ab)
        # for the sake of completeness, also set one-orbital densities outside the active space
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
        rdm1a, rdm1b = self.rdm1s(root=root)
        rdm2 = self.rdm2(root=root)
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

    def rdm1s(self, root: int = 0) -> np.ndarray:
        """Returns the active one-electron spin density matrix as an 2 x N x N array."""
        if not self.casci:
            raise RuntimeError("Object has no attribute with CASCI results.")

        # Number of active orbitals and electrons.
        ncas = self.casci.ncas
        nel = self.casci.nelecas

        # Density matrices are computed directly by the full CI solver object.
        fcisolver = self.casci.fcisolver

        # CI coefficients are stored as a vector or a list of vectors, depending on the number of
        # roots requested.
        civec = fcisolver.ci if fcisolver.nstates == 1 else fcisolver.ci[root]

        # Obtain the one-electron spin-density matrix with its alpha and beta components.
        rdm1a, rdm1b = fcisolver.make_rdm1s(civec, ncas, nel)
        return np.array([rdm1a, rdm1b])

    def rdm2(self, root: int = 0) -> np.ndarray:
        """Returns the active two-electron spin-free density matrix as a four-dimensional array."""
        if not self.casci:
            raise RuntimeError("Object has no attribute with CASCI results.")

        # Number of active orbitals and electrons.
        ncas = self.casci.ncas
        nel = self.casci.nelecas

        # Density matrices are computed directly by the full CI solver object.
        fcisolver = self.casci.fcisolver

        # CI coefficients are stored as a vector or a list of vectors, depending on the number of
        # roots requested.
        civec = fcisolver.ci if fcisolver.nstates == 1 else fcisolver.ci[root]

        # Obtain the two-electron spin-free density matrix.
        return np.array(fcisolver.make_rdm2(civec, ncas, nel))
