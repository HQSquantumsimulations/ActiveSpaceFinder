# Copyright © 2024 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Procedures for determining an initial active space."""


from typing import Optional, Union

import numpy as np
from pyscf.dft.rks import KohnShamDFT
from pyscf.gto import Mole
from pyscf.mp.dfmp2_native import DFRMP2
from pyscf.mp.dfump2_native import DFUMP2
from pyscf.scf.hf import RHF as RHFclass
from pyscf.scf.hf import SCF as SCFclass
from pyscf.scf.uhf import UHF as UHFclass

from .asfbase import ActiveSpace, Preselection
from .natorbs import select_natural_occupations
from .rimp2_pairinfo import corresponding_orbital_subspaces, diag_cumulant_ump2
from .scf import SCFtype


class MP2NatorbPreselection(Preselection):
    """Preselection strategy based on natural occupations between a lower and an upper boundary.

    Given a converged Hartree-Fock solution, this procedure calculates MP2 natural orbitals (from
    the orbital-unrelaxed density matrix). An active space is selected by including all natural
    orbitals with associated eigenvalues between a lower and an upper threshold. In addition,
    boundaries are imposed to select no less than a minimum and no more than a maximum number of
    orbitals for the active space.

    Attributes:
        lower:    lower boundary for the natural occupation numbers
        upper:    upper boundary for the natural occupation numbers
        min_orb:  minimal number of orbitals
        max_orb:  maximal number of orbitals
        scf:      SCF object (RHF or UHF)
        scf_type: the SCF object's SCF type
    """

    def __init__(
        self,
        scf: SCFclass,
        lower: float = 0.01,
        upper: float = 1.99,
        min_orb: Optional[int] = 10,
        max_orb: Optional[int] = 30,
    ) -> None:
        """Initialize the preselection procedure.

        Args:
            scf:      SCF object (RHF or UHF)
            lower:    lower boundary for the natural occupation numbers
            upper:    upper boundary for the natural occupation numbers
            min_orb:  minimal number of orbitals
            max_orb:  maximal number of orbitals
        """
        self.lower = lower
        self.upper = upper
        self.min_orb = min_orb
        self.max_orb = max_orb
        self.scf = scf
        self.scf_type = self._determine_scf_type(self.scf)

    @property
    def mol(self) -> Mole:  # noqa: D102
        return self.scf.mol

    def _determine_scf_type(self, scf: SCFclass) -> SCFtype:
        """Determine the SCF type of an SCF object.

        Args:
            scf: SCF object (RHF or UHF)

        Returns:
            SCF type (if successful)

        Raises:
            TypeError: invalid SCF object
        """
        if isinstance(scf, RHFclass):
            return SCFtype.RHF
        elif isinstance(scf, UHFclass):
            return SCFtype.UHF
        elif isinstance(scf, KohnShamDFT):
            raise TypeError("DFT objects are not supported.")
        else:
            raise TypeError("scf must be a RHF or UHF object.")

    def compute_natorbs(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute MP2 natural orbitals.

        The density-fitting MP2 implementation (RHF-DF-MP2 or UHF-DF-MP2) will be used to
        compute the natural orbitals.

        Returns:
            natural occupation numbers, natural orbitals

        Raises:
            ValueError: bad natural occupation numbers
        """
        natocc, natorb = np.array([]), np.array([])
        if self.scf_type == SCFtype.RHF:
            natocc, natorb = DFRMP2(self.scf).make_natorbs()
        elif self.scf_type == SCFtype.UHF:
            natocc, natorb = DFUMP2(self.scf).make_natorbs()

        if np.any(np.diff(natocc) > 0.0):
            raise ValueError("Natural occupation numbers are not in descending order.")
        return natocc, natorb

    def select(self) -> ActiveSpace:
        """Perform preselection based on MP2 natural orbital occupation numbers.

        Returns:
            An active space based on MP2 natural orbitals.
        """
        natocc, natorb = self.compute_natorbs()
        nel, mo_list = select_natural_occupations(
            natocc=natocc,
            lower=self.lower,
            upper=self.upper,
            min_orb=self.min_orb,
            max_orb=self.max_orb,
        )
        return ActiveSpace(nel=nel, mo_list=mo_list, mo_coeff=natorb)


class MP2PairinfoPreselection(Preselection):
    """Preselection strategy based on pair information from perturbation theory.

    The procedure generates an initial active space of orbitals similar to MP2 natural orbitals.

    Attributes:
        reference:      Either a UHF (a DFUMP2 object is constructed) or an existing DFUMP2 object.
        pair_thresh:    Cutoff to truncate pair information.
        svd_thresh:     Cutoff to construct a minimal active space from UHF corresponding orbitals.
        active_mp2no:   Orbitals in minimal active space are MP2 natural orbital-like.
        inactive_mp2no: All other orbitals are MP2 natural orbital-like.
    """

    def __init__(
        self,
        reference: Union[UHFclass, DFUMP2],
        pair_thresh: float = 2.0e-3,
        svd_thresh: float = 0.98,
        active_mp2no: bool = True,
        inactive_mp2no: bool = True,
    ) -> None:
        """Initialize the preselection procedure.

        Args:
            reference:      Either a UHF (a DFUMP2 object is constructed) or an existing DFUMP2
                            object.
            pair_thresh:    Cutoff to truncate pair information.
            svd_thresh:     Cutoff to construct a minimal active space from UHF corresponding
                            orbitals.
            active_mp2no:   Orbitals in minimal active space are MP2 natural orbital-like.
            inactive_mp2no: All other orbitals are MP2 natural orbital-like.

        Raises:
            TypeError: invalid input type
        """
        self.pair_thresh = pair_thresh
        self.svd_thresh = svd_thresh
        self.active_mp2no = active_mp2no
        self.inactive_mp2no = inactive_mp2no
        if isinstance(reference, UHFclass):
            self.pt = DFUMP2(reference)
        elif isinstance(reference, DFUMP2):
            self.pt = reference
        else:
            raise TypeError("'reference' must be either of UHF or DFUMP2 type.")

    @property
    def mol(self) -> Mole:  # noqa: D102
        return self.pt._scf.mol

    def select(self) -> ActiveSpace:
        """Perform preselection based on RI-MP(2, 1) pair information.

        Returns:
            An initial active space of orbitals similar to MP2 natural orbitals.
        """
        # Construct subspaces: minimal MO space, unrestricted and restricted MO coefficients.
        mos_minimal, mo_coeff_ur, mo_coeff_re = corresponding_orbital_subspaces(
            pt=self.pt,
            threshold=self.svd_thresh,
            active_mp2no=self.active_mp2no,
            inactive_mp2no=self.inactive_mp2no,
        )

        # Identify cumulant elements above the specified threshold.
        cumulant = diag_cumulant_ump2(self.pt, mo_coeff_ur)
        above_thresh = np.argwhere(np.abs(cumulant) >= self.pair_thresh)

        # above_thresh is an N x 3 array. Each row contains indices spin, MO 1, MO 2
        # The spin is ignored: there is a 1-1 correspondence between unrestricted MOs, so the
        # indices are taken as they are.
        mo_set = set(mos_minimal)
        mo_set.update(np.ravel(above_thresh[:, 1:]))
        mo_list = sorted(mo_set)

        # Counting the electrons: the ordering of MOs is occupied - active (UNOs) - virtual.
        # Provided that the svd_thresh is chosen to capture most of the symmetry breaking,
        # the number of electrons can be determined using the SCF occupations.
        occ_a = np.intersect1d(np.arange(self.pt.nocc[0]), mo_list)
        occ_b = np.intersect1d(np.arange(self.pt.nocc[1]), mo_list)
        nel = len(occ_a) + len(occ_b)

        return ActiveSpace(nel=nel, mo_list=mo_list, mo_coeff=mo_coeff_re)
