from typing import List, Tuple, Union

import numpy as np

from pyscf.mcscf import CASCI
from pyscf.gto import Mole

from .asf import calcS1, selectOrbitals, default_entropy_threshold, default_plateau_threshold


class ASFCI:
    """
    Sets up and performs CASCI calculations for entropy-based active space selection.
    """

    def __init__(self,
                 mol: Mole,
                 mo_coeff: np.ndarray,
                 nel: int = None,
                 norb: int = None) -> None:
        """
        Sets initial parameters for a CASCI calculation with a specific number of
        electrons and orbitals. Without giving a number of electrons and orbitals,
        it initializes a full CI calculation.

        Args:
            mol: molecular information
            mo_coeff: spin-restricted molecular orbital coefficients
            nel: number of electrons in the initial orbital space
            norb: number of orbitals in the initial space
        """
        self.mol = mol
        self.mo_coeff = mo_coeff

        if nel is None:
            self.nel = mol.nelectron
        else:
            self.nel = nel

        if norb is None:
            self.norb = mo_coeff.shape[1]
        else:
            self.norb = norb

        self.mo_list: Union[List[int], None] = None
        self.casci: Union[CASCI, None] = None

    def setMOList(self, nel: int, mo_list: List[int]) -> None:
        """
        Detailed specification of an initial orbital space, from which the active orbitals
        will be selected subsequently. Note: mo_list uses base 0 (in contrast to PySCF's mo_sort
        function, which, counterintuitively, uses base 1 by default). Overrides the settings
        provided in __init__.

        Args:
            nel: number of electrons in the initial orbital space
            mo_list: list of MO indices for the initial orbital space
        """
        # TODO: need to make sure that entropies map back on the original orbitals...
        self.nel = nel
        self.mo_list = mo_list
        self.norb = len(mo_list)

    def runCI(self, spin_shift: float = None, **kwargs) -> None:
        """
        Run the (CAS-)CI calculation for subsequent orbital selection.

        Args:
            spin_shift: optionally, the shift for PySCF's fix_spin function
            **kwargs: arguments that can be passed on the the FCI solver
        """
        casci = CASCI(self.mol, self.norb, self.nel)
        if spin_shift is None:
            casci.fix_spin()
        else:
            casci.fix_spin(shift=spin_shift)

        for key, value in kwargs.items():
            setattr(casci.fcisolver, key, value)

        if self.mo_list is None:
            casci_mos = self.mo_coeff
            self.mo_list = list(range(casci.ncore, casci.ncore + casci.ncas))
        else:
            casci_mos = casci.sort_mo(caslst=self.mo_list, mo_coeff=self.mo_coeff, base=0)
        casci.kernel(mo_coeff=casci_mos)

        self.casci = casci

    def selectOrbitals(self,
                       threshold: float = default_entropy_threshold,
                       plateau_threshold: float = default_plateau_threshold,
                       map_MOs: bool = True,
                       verbose: bool = True) -> Tuple[int, List[int]]:
        """
        Performs the orbital selection.

        Args:
            threshold: threshold for the single-site entropies
            plateau_threshold: threshold to identify plateaus
            map_MOs: set whether to map the MO indices back to the original order
                     relevant if a custom mo_list was specified
            verbose: print information or not

        Returns:
            number of electrons, list of selected orbitals
        """
        assert(self.casci is not None)
        assert(self.mo_list is not None)

        orbdens = self.getOrbDens()
        entropy = self.calcEntropy(orbdens)
        nel, mos = selectOrbitals(orbdens, threshold=threshold,
                                          plateau_threshold=plateau_threshold)

        if map_MOs:
            mos_mapped = []
            for i in mos:
                idx_act = i - self.casci.ncore
                idx_orig = self.mo_list[idx_act]
                mos_mapped.append((idx_orig))
        else:
            mos_mapped = mos

        if verbose:
            print(' MO    w( )    w(\u2191)    w(\u2193)    w(\u21c5)       S  sel')
            for i in range(self.casci.ncas):
                idx_act = self.casci.ncore + i
                if map_MOs:
                    idx_mapped = self.mo_list[i]
                else:
                    idx_mapped = idx_act
                if idx_act in mos:
                    status = 'a'
                else:
                    status = ' '

                print('{0:3d}  {1:6.3f}  {2:6.3f}  {3:6.3f}  {4:6.3f}  {5:6.3f}  {6:>3s}'.
                      format(idx_mapped, orbdens[idx_act, 0], orbdens[idx_act, 1],
                             orbdens[idx_act, 2], orbdens[idx_act, 3], entropy[idx_act], status))

        return nel, mos_mapped

    def getOrbDens(self, root: int = 0) -> np.ndarray:
        """
        Determines the one-orbital density for the previously calculated (CAS-)CI wave function.

        Args:
            root: state that the density is calculated for

        Returns:
            The one-orbital density for all orbitals as an N x 4 numpy array.
            Order of the columns is empty, spin-up, spin-down, doubly occupied.
        """
        assert(self.casci is not None)

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

        assert(rdm1a.shape == (ncas, ncas))
        assert(rdm1b.shape == (ncas, ncas))
        assert(rdm2ab.shape == (ncas, ncas, ncas, ncas))

        orbdens = np.zeros((nmo, 4))

        # for the sake of completeness, also set one-orbital densities outside the active space
        orbdens[:ncore, 3] = 1.0
        orbdens[(ncore + ncas):, 0] = 1.0

        # the relevant part: one-orbital density inside the active space
        for i in range(ncas):
            rdm2s = rdm2ab[i, i, i, i]
            orbdens[ncore + i, 0] = 1.0 - rdm1a[i, i] - rdm1b[i, i] + rdm2s
            orbdens[ncore + i, 1] = rdm1a[i, i] - rdm2s
            orbdens[ncore + i, 2] = rdm1b[i, i] - rdm2s
            orbdens[ncore + i, 3] = rdm2s

        return orbdens

    def calcEntropy(self,
                    orbdens: np.ndarray = None,
                    root: int = 0) -> np.ndarray:
        """
        Calculates the one-site entropy for a (CAS-)CI wave function.

        Args:
            orbdens: optionally the orbital density to calculate the entropy for
            root: the root that the one-orbital density is calculated for,
                  if no density is provided

        Returns:
            list of single-site entropies for the orbitals
        """
        if orbdens is None:
            orbdens = self.getOrbDens(root=root)
        return calcS1(orbdens)
