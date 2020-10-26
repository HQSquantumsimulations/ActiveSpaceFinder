# Copyright 2020 HQS Quantum Simulations GmbH
# Reza Ghafarian Shirazi, Thilo Mast.
# reza.shirazi@quantumsimulations.de
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################
#    Libraries
################################
import os
import numpy as np
from pyscf import scf, mcscf, dmrgscf, gto
import timeit
from typing import Union, Optional, List
from asf import orbs_el_n, act_fao, plot_selection, visualize_mos, save_csv
from asf import AS_true, scf_mo_occ, del_DMRG_temp_, AS_compare
from asf import nat_orbs_rdm1AO_rhf, nat_orbs_obj
# CASSCF CASCI classes
from pyscf.mcscf.casci import CASCI as CASCIClass
from pyscf.mcscf.mc1step import CASSCF as CASSCFClass

################################
#    Functions
################################


class asf:

    def __init__(self, verbose: int = None, DMRG_M: Optional[Union[int, List[int]]] = None,
                 DMRG_S: Optional[Union[int, list]] = None,
                 DMRG_tol: Optional[Union[float, list]] = None,
                 DMRG_N: Optional[Union[float, list]] = None, DMRG_MaxM: int = None,
                 DMRG_SM: Optional[int] = None, tdod: int = None, restart: bool = None,
                 nthreads: int = None, entropies: list = None, run_dmrg_flag: bool = None,
                 fontsize: float = None, fontsize2: float = None, linewidth: float = None,
                 loc: str = None, fancybox: bool = None, default_color: str = None,
                 tolerance: float = None, mc: Union[CASCIClass, CASSCFClass] = None,
                 no: list = None, occ: list = None, memory: int = None, dbug: bool = False,
                 dcolor: str = None, filename_nat: str = '', filename_ent: str = '',
                 idxshift=None, p_thresh: Optional[List[dict]] = None):
        r"""Set general attributes.

            Args:
                verbose (int):              Print level.
                DMRG_M (int/lsit):          Provide a custom number of M for DMRG calculation.
                DMRG_S (int/list):          Number of DMRG sweeps.
                DMRG_tol (float/list):      DMRG schedule tol.
                DMRG_N (float/list):        DMRG schedule noise.
                DMRG_MaxM (int):            Maximum value of M in sweeps.
                                            Default 251.
                DMRG_SM (int):              DMRG startM.
                tdod (int):                 DMRG twodot_to_onedot.
                restart (bool):             Block restart flag.
                nthreads (int):             Number of cores to use for the DMRG calculation.
                entropies (list):           List of previosly calculated entropies.
                                            Entering this will skip DMRG calculations.
                run_dmrg_flag (bool):       Flag for running DMRG or resuming with the entropies.
                fontsize (float):           Font size in plotting.
                fontsize2 (float):          Fontsize for legend.
                linewidth (float):          Line width for threshold lines.
                loc (str):                  Placement of plot legend.
                fancybox (bool):            FancyBox option for legend.
                default_color (str):        Color for mathplotlob. Changes the default
                                            threshold line color.
                tolerance (float):          General tolerance for 0 / Machine 0.
                mc (CASCIClass):            PySCF/DMRG MCSCF object.
                no (list):                  Natural orbitals from DMRG.
                occ (list):                 Natural occupations from DMRG.
                memory (int):               DMRG memory in MB.
                                            Default is grabbed from mol.
                dbug (bool):                debug flag.
                dcolor (str):               Color of selected criteria.
                                            For instance the color of selected threshold
                                            in plotting.
                filename_nat (str):         Filename for saving natural orbital occupations.
                filename_ent (str):         Filename for saving entropy file.
                idxshift (int):             Shift idx of x axis.
                p_thresh (list):            thresholds.

            Raises:
                TypeError: read the error.

        """
        # Defaults
        self.p_thresh: List[dict]
        if p_thresh is not None:
            self.p_thresh = p_thresh
        else:
            self.p_thresh = []

        if dbug is not None:
            self.dbug = dbug
        else:
            self.dbug = None

        # printing
        if verbose is None:
            self.verbose = 3
        else:
            self.verbose = verbose

        # DMRG settings
        if DMRG_M is not None:
            if isinstance(DMRG_M, list):
                self.DMRG_M = DMRG_M
            elif isinstance(DMRG_M, int):
                self.DMRG_M = [DMRG_M]
            else:
                raise TypeError('DMRG_M should be a list or int')
        else:
            self.DMRG_M = []

        if DMRG_S is not None:
            if isinstance(DMRG_S, list):
                self.DMRG_S = DMRG_S
            elif isinstance(DMRG_S, int):
                self.DMRG_S = [DMRG_S]
            else:
                raise TypeError('DMRG_S should be a list or int')
        else:
            self.DMRG_S = []

        if DMRG_tol is not None:
            if isinstance(DMRG_tol, list):
                self.DMRG_tol = DMRG_tol
            elif isinstance(DMRG_tol, float):
                self.DMRG_tol = [DMRG_tol]
            else:
                raise TypeError('DMRG_tol should be float or list')
        else:
            self.DMRG_tol = []

        if DMRG_N is not None:
            if isinstance(DMRG_N, list):
                self.DMRG_N = DMRG_N
            elif isinstance(DMRG_N, float):
                self.DMRG_N = [DMRG_N]
            else:
                raise TypeError('DMRG_N should be float or list')
        else:
            self.DMRG_N = []

        self.tdod: Optional[int]
        if tdod is not None:
            self.tdod = tdod
        else:
            self.tdod = None

        if DMRG_MaxM is not None:
            self.DMRG_MaxM = DMRG_MaxM
        else:
            self.DMRG_MaxM = 251

        self.DMRG_SM: Optional[int]
        if DMRG_SM is not None:
            if isinstance(DMRG_SM, int):
                self.DMRG_SM = DMRG_SM
            else:
                raise TypeError('DMRG_SM should be an int')
        else:
            self.DMRG_SM = None

        if restart is not None:
            self.restart = restart
        else:
            self.restart = False

        self.nthreads: Optional[int]
        if nthreads is not None:
            self.nthreads = nthreads
        else:
            self.nthreads = None
        if memory is None:
            self.memory = None
        else:
            self.memory = memory

        # Plots
        if fontsize is not None:
            self.fontsize = fontsize
        else:
            self.fontsize = 9.0
        if fontsize2 is not None:
            self.fontsize2 = fontsize2
        else:
            self.fontsize2 = 8.0
        if linewidth is not None:
            self.linewidth = linewidth
        else:
            self.linewidth = 1.1
        if fancybox is not None:
            self.fancybox = fancybox
        else:
            self.fancybox = True
        if idxshift is not None:
            self.idxshift = idxshift
        else:
            self.idxshift = 0

        # DMRG Entropies
        self.entropies: List[List[float]]
        if entropies is not None:
            self.entropies = entropies
        else:
            self.entropies = []

        # Resume
        if run_dmrg_flag is not None:
            self.run_dmrg_flag = run_dmrg_flag
        else:
            self.run_dmrg_flag = False

        # computation
        if tolerance is None:
            self.tolerance = 1.0e-12
        else:
            self.tolerance = tolerance

        # DMRG results
        if mc is None:
            self.mc = None
        else:
            self.mc = mc

        if occ is not None:
            self.occ = occ
        else:
            self.occ = []

        if no is not None:
            self.no = no
        else:
            no = []

        # etc
        if dcolor is None:
            self.dcolor = 'RED'
        else:
            self.dcolor = dcolor

        if not filename_ent:
            self.filename_ent = ''
        else:
            self.filename_ent = filename_ent

        if not filename_nat:
            self.filename_nat = ''
        else:
            self.filename_nat = filename_nat

    def select_mroot_AS(self, mos: list, base: int = 0) -> list:
        r"""Selection of active space based on several active spaces \*(roots).

            Args:
                mos (list):     List of lists of active spaces.
                                idx should be starting from 0.
                base (int):     Start counting orbitals from 0.
                                Change to 1 if you want to feed this
                                directyly to sort_mo functinon of pyscf.

            Returns:
                mos (list):     The molecular orbitals of the final active space.

            Note:
                Assuming the calculations have been performed on the same active space
                size. All orbitals in all calculations should be identical or
                at least very similar.
                """

        if AS_compare(mos) is True:
            mos = mos[0]
            # If the active spaces don't match
        else:
            # Final active space will be the union of all active spaces in n number of roots.
            mos_u = mos[0]
            # i is number of roots
            for i in range(1, len(mos)):
                mos_u = list(set(mos_u) | set(mos[i]))


        # Convert to numpy array
        mos_a = np.array(mos_u)

        mos_a.sort()
        if base != 0:
            base = int(np.round(base))
            mos = np.array(mos_a + base)
        else:
            mos = mos_a

        if self.dbug:
            print("")
            print('base :', base)
            print('MOs:', len(mos), ", ", mos)
            print("")

        return list(mos)

    def reduce_orbs_n(self, in_orbs: list, nat_occ: list, score: np.ndarray,
                      machine_limit: int, max_loop: int = 1, incr_per: float = 0.1):
        r"""Reducing the size of the active space based on occupation numbers.
            Do not use the loop feature in this Version!

            Args:
                in_orbs (list):             Initial list of orbitals to be reduced.
                nat_occ (list):             Natural orbital occupation numbers.
                score (list):               The score each orbital made based on the threshold
                                            values.
                machine_limit (int):        Maximum allowed selection.
                max_loop (int):             Howmany times loop to achieve enough DOMOs.
                                            Set to 1 for no loop.
                incr_per (float):           Percent increase in DOMO values.

            Returns:
                Chosen_ones (list):         A new final active space with the number of orbitals
                                            as machine limit.
        """

        q_act_s = False
        flag_corr_o = False
        flag_corr_v = False
        loop = 0

        orb_occ = np.round(nat_occ)
        if orb_occ.ndim > 1:
            orb_occ = np.add(orb_occ[0, :], orb_occ[1, :], type=int)
        else:
            orb_occ = orb_occ

        while q_act_s is False:
            Chosen_ones = []
            # Sort based on the calculated score
            score = np.array(score)
            # Reverse larger is better
            Ch_sorted = score[score[:, 1].argsort()[::-1]]
            loop += 1
            # loop on the sorted score until the limit is reached
            for i in Ch_sorted[:, 0]:
                Ch_sorted_i = i
                Chosen_ones.append(int(Ch_sorted_i))
                if len(Chosen_ones) > machine_limit - 1:
                    # reorder
                    Chosen_ones.sort()
                    if self.dbug:
                        print("")
                        print("selected orbitals reduced to :", len(Chosen_ones), ",",
                              Chosen_ones)
                        print("DOMOs:", orbs_el_n(orb_occ, Chosen_ones, cshell=True) / 2,
                              'SOMOs: ', orbs_el_n(orb_occ, Chosen_ones) - orbs_el_n(orb_occ,
                              Chosen_ones, cshell=True), 'of total: ',
                              int(len(Chosen_ones)))
                        print("selected electrons :", orbs_el_n(orb_occ, Chosen_ones))
                        print("")
                    break

            # This is an alpha implementation not tested!
            # We check the number of SOMOs/Vir and DOMOs try to avoid selecting
            # too DOMOs or SOMOs/Virt in relation to one another. Current limit
            # is arbitrary.

            # Check number of DOMOs
            if orbs_el_n(orb_occ, Chosen_ones, cshell=True) / 2 < len(Chosen_ones) / 3:
                if self.dbug:
                    print("Warning too many empty too few DOMOs:",
                          orbs_el_n(orb_occ, Chosen_ones, cshell=True) / 2, 'of: ',
                          int(len(Chosen_ones)), 'loop:', loop)
                    print("Looping again with weights favoring DOMOs")
                print("")

                flag_corr_o = True

                for idx, i in enumerate(score[:, 0]):
                    i = int(i)
                    if int(np.round(orb_occ[i])) == 2:
                        score[idx, 1] = score[idx, 1] + score[idx, 1] * incr_per

            # Check number of Vir/SOMOs
            elif len(Chosen_ones) - orbs_el_n(orb_occ, Chosen_ones, cshell=True) / 2 < \
                    orbs_el_n(orb_occ, Chosen_ones, cshell=True) / 4:
                if self.dbug:
                    print("Warning too many empty too few Virs/SOMOs:", orbs_el_n(orb_occ,
                          Chosen_ones, cshell=True) / 2, 'of: ',
                          int(len(Chosen_ones)), 'loop:', loop)
                    print("Looping again with weights favoring DOMOs")
                    print("")

                flag_corr_v = True

                for idx, i in enumerate(score[:, 0]):
                    i = int(i)
                    if int(np.round(orb_occ[i])) == 0:
                        score[idx, 1] = score[idx, 1] + score[idx, 1] * incr_per

            # If All is good quit loop with debug flags
            else:
                # If DOMO was being corrected
                if flag_corr_o is True:
                    if self.dbug:
                        print("Success at loop:", loop, "DOMOs:", orbs_el_n(orb_occ, Chosen_ones,
                              cshell=True) / 2, 'of: ', int(len(Chosen_ones)))
                        print("")
                # If SOMO/Vir was being corrected
                if flag_corr_v is True:
                    if self.dbug:
                        print("Success at loop:", loop, "Virs/SOMOs:",
                              len(Chosen_ones) - orbs_el_n(orb_occ, Chosen_ones, cshell=True) / 2,
                              'of: ', int(len(Chosen_ones)))
                        print("")
                # Set the while flag to quit.
                q_act_s = True

            # Max iterations reached.
            if loop >= max_loop:
                if self.dbug:
                    print("Max loop reached!")
                    print("")
                break

        return Chosen_ones

    def calculate_criteria(self, nat_occ: list, weight: bool = False,
                           we_sc: float = 4, clipping: bool = True, Th_auto: str = 'HQS'):
        r"""automatic selection of the threshold. The automatgic generation
            of threshold is different for occupied and virtual (empty) orbitals. Singly
            occupied threshold is set as occupied, however, they are always selected
            unless you switch singles to off.

            Args:
                nat_occ (list):             Natural orbital occupation numbers.
                weight (bool):  		    Factor in weights of occupied and unoccupied orbitals in
                                            threshold calculation.
                we_sc (float):              The scale to which weight is considered.
                                            Check the equation below.
                clipping (bool):            Clipping the very small values from the occupation list.
                                            This helps with setting a proper threshold.
                Th_auto (str):              The choices of automated threshold selection.
                                            'HQS' default.
                                            'STD' stanorredard deviation is used.
                                            'AVG' average value is used.
                                            'Min' Minimum of above.
                                            'Max' Max of above.

            return:
                crit (dict):                A dictionary for containing calculated criteria's.

            raises:
                TypeError: look up the error.
        """
        # Deviation from pure occupations [2,1,0]
        total_occ = np.round(nat_occ)
        dif = np.absolute(nat_occ - total_occ)

        # Numver of doubly, virtual and singly occupied orbitals
        n_docc = ((1.99 < total_occ) & (total_occ < 2.01)).sum()
        n_empt = ((0.0 <= total_occ) & (total_occ < 0.01)).sum()
        n_sing = ((0.99 <= total_occ) & (total_occ < 1.01)).sum()

        dif_occ = dif[0:n_docc + n_sing]
        dif_vir = dif[n_docc + n_sing:]

        # Make sure everything is correct
        if not len(dif_occ) == n_sing + n_docc:
            raise TypeError('Fatal error.')
        if not len(dif_vir) == n_empt:
            raise TypeError('Fatal error.')
        if not n_docc + n_empt + n_sing == len(total_occ):
            raise TypeError('Fatal error.')
        # The weights

        # Uncommented because, not used
        # torbs = len(nat_occ)
        # n_sing = torbs - n_docc - n_empt
        wdocc = n_docc / n_empt
        wempt = n_empt / n_docc

        # Clipping the very small values from the list
        # We look through the weight of empty orbitals
        # to find the small value to cut from
        # more appropriate for large basis set organics
        # Rationale is that the very small values bring down
        # the threshold and in large basis sets there are
        # many empty orbitals with very small occupations.
        # The weight distribution helps dealing with such
        # large basis sets.

        if clipping is True:
            if wempt < 1:
                clipped_dif = dif_vir
            if wempt < 2:
                clipped_dif = dif_vir[(0.0007 < dif_vir)]
            elif wempt < 4:
                clipped_dif = dif_vir[(0.0025 < dif_vir)]
            elif wempt >= 4:
                clipped_dif = dif_vir[(0.0045 < dif_vir)]
            elif wempt >= 6:
                clipped_dif = dif_vir[(0.0065 < dif_vir)]
            elif wempt >= 10:
                clipped_dif = dif_vir[(0.0077 < dif_vir)]
            elif wempt >= 12:
                clipped_dif = dif_vir[(0.0083 < dif_vir)]
        else:
            clipped_dif = dif_vir

        # Automated value for external orbitals

        HQS = sum((clipped_dif - np.mean(clipped_dif)**2) / (len(clipped_dif) - 1))
        STD = np.std(clipped_dif, axis=0)
        AVG = np.average(clipped_dif, axis=0)

        if self.dbug:
            print("External automated thresholds :::", Th_auto)
            print("HQS :", HQS, " STD :", STD, " AVG :", AVG)

        if Th_auto.upper() == 'HQS':
            value_vir = HQS
        elif Th_auto.upper() == 'STD':
            value_vir = STD
        elif Th_auto.upper() == 'AVG':
            value_vir = AVG
        elif Th_auto.upper() == 'MAX':
            value_vir = max(HQS, AVG, STD)
        elif Th_auto.upper() == 'MIN':
            value_vir = min(HQS, AVG, STD)
        else:
            print('Th_auto unknown:', Th_auto)

        if wdocc < 0.2:
            clipped_dif = dif_occ[(0.0000 < dif_occ)]
        if wdocc < 0.5:
            clipped_dif = dif_occ[(0.0001 < dif_occ)]
        elif wdocc < 0.95:
            clipped_dif = dif_occ[(0.00015 < dif_occ)]
        elif wdocc >= 0.95:
            clipped_dif = dif_occ[(0.0002 < dif_occ)]

        # Automated value for internal orbitals

        HQS = sum((clipped_dif - np.mean(clipped_dif)**2) / (len(clipped_dif) - 1))
        STD = np.std(clipped_dif, axis=0)
        AVG = np.average(clipped_dif, axis=0)

        if self.dbug:
            print("Internal automated thresholds :::", Th_auto)
            print("HQS :", HQS, " STD :", STD, " AVG :", AVG)

        if Th_auto.upper() == 'HQS':
            value_occ = HQS
        elif Th_auto.upper() == 'STD':
            value_occ = STD
        elif Th_auto.upper() == 'AVG':
            value_occ = AVG
        elif Th_auto.upper() == 'MAX':
            value_occ = max(HQS, AVG, STD)
        elif Th_auto.upper() == 'MIN':
            value_occ = min(HQS, AVG, STD)
        else:
            print('Th_auto unknown:', Th_auto)

        # prepare the dictionary
        crit = {}

        # Threshold setting loop
        for key in ['lost_occ', 'gained_occ', 'gain_loss']:
            if key == "gained_occ":
                if weight is True:
                    crit[key] = (we_sc * value_vir + value_vir / wempt) / we_sc
                else:
                    crit[key] = value_vir
            elif key == "lost_occ":
                if weight is True:
                    crit[key] = (we_sc * value_occ + value_occ / wdocc) / we_sc
                else:
                    crit[key] = value_occ
            else:
                crit[key] = value_occ

        crit['singles'] = 1

        return crit

    def select_mos_occ(self, nat_occ: list, crit: dict = None, plot: bool = False,
                       weight: bool = False, we_sc: float = 4,
                       ptitle: str = 'Natural occupation', machine_limit: int = 30,
                       max_loop: int = 1, incr_per: float = 0.05,
                       clipping: bool = True, Th_auto: str = 'HQS', loop: int = 1):
        r"""Performs a selection of orbitals based on natural orbital occupations.
            The criteria for selection is calculated automatically using calculate_criteria
            function. If you wish to define your own thresholds use crit.

            Args:
                nat_occ (list):             Natural orbital occupation numbers.
                crit (dict):                The criteria for orbital selection.
                                            If set to None this is calculated automatically
                                            using a method based on one of the automatic methods.
                                            Otherwise one can set the criteria manually
                                            in a flexible way:
                                            'lost_occ' is the occupation lost by doubly occ. orbs.
                                            'gained_occ' is the occupation gained by empty orbs.
                                            'gain_loss' is the occupation change by singly occupied
                                            orbs.
                                            'singles' select all singly occupied orbitals
                                            crit={
                                            'lost_occ':0.1,
                                            'gained_occ':0.1,
                                            'gain_loss':0.1,
                                            'singles':1}
                plot (bool):                To plot Natural orbital selection.
                ptitle (str):               Title of the plot.
                weight (bool):  		    Factor in weights of occupied and unoccupied orbitals in
                                            threshold calculation.
                                            For criteria function only.
                we_sc (float):              The scale to which weight is considered.
                                            Check the equation below.
                                            For criteria function only.
                machine_limit (int):        Maximum allowed selection.
                max_loop (int):             Howmany times loop to achieve enough DOMOs.
                                            for reduction function only.
                incr_per (float):           Percent increase in DOMO values.
                                            for reduction function only.
                clipping (bool):            Clipping the very small values from the occupation list.
                                            This helps with setting a proper threshold.
                                            For criteria function only.
                Th_auto (str):              The choices of automated threshold selection.
                                            'HQS' default.
                                            'STD' stanorredard deviation is used.
                                            'AVG' average value is used (recommended for closed
                                            shell systems).
                                            'Min' Minimum of above.
                                            'Max' Max of above.
                                            To disable the feature assign your choice of crit.
                                            for selecting crit see up.
                                            For criteria function only.
                loop (int):                 Loop iteration for appropriation balance of empty
                                            and occupied orbitals. Please, do not use this feature
                                            in this version.
                                            For reduction function only.

            Returns:
                Chosen_ones (np.ndarray):   List of selected orbitals.

            Self:
                diff (np.ndarray):          Occupation difference with HF.
                crit (dict):                Automatically selected threshold.

            Raises:
                TypeError: look up error.
                Exception: look up error.
        """

        if not isinstance(nat_occ, (np.ndarray, list)):
            raise TypeError('occuptation should be inputted as a numpy array or list')

        nat_occ = list(nat_occ)
        # Deviation from pure occupations [2,1,0]
        total_occ = np.round(nat_occ)
        dif = np.absolute(nat_occ - total_occ)

        # Numver of doubly, virtual and singly occupied orbitals
        n_docc = ((1.99 < total_occ) & (total_occ < 2.01)).sum()
        n_empt = ((0.0 <= total_occ) & (total_occ < 0.01)).sum()
        n_sing = ((0.99 <= total_occ) & (total_occ < 1.01)).sum()

        dif_occ = dif[0:n_docc + n_sing]
        dif_vir = dif[n_docc + n_sing:]

        # Make sure everything is correct
        if not len(dif_occ) == n_sing + n_docc:
            raise TypeError('Fatal Error.')
        if not len(dif_vir) == n_empt:
            raise TypeError('Fatal Error.')
        if not n_docc + n_empt + n_sing == len(total_occ):
            raise TypeError('atal Error.')

        # Uncommented because, not used
        # torbs = len(nat_occ)
        # n_sing = torbs - n_docc - n_empt
        wdocc = n_docc / n_empt
        wempt = n_empt / n_docc

        # Automatic Threshold selection
        if crit is None:
            crit = self.calculate_criteria(nat_occ, weight=weight, we_sc=we_sc, clipping=clipping,
                                           Th_auto=Th_auto)

        # Make sure thresholds are in a dic
        if not isinstance(crit, dict):
            raise TypeError('crit should be a dictionary.')

        if self.dbug:
            print("Natural occupation threshold set to :::", crit)
            if crit is False:
                print("weights are (occupied, empty) :::", wdocc, wempt)

        # Orbital selection based on the criteria
        # Selected orbitals list
        Chosen_ones = []
        # Score which will be used for further reduction if necessary
        score = []

        # Selection loop
        for i in range(len(nat_occ)):
            if abs(total_occ[i]) < self.tolerance:
                if dif[i] > crit['gained_occ']:
                    score.append((i, dif[i] - crit['gained_occ']))
                    Chosen_ones.append(i)

            if abs(total_occ[i] - 1) < self.tolerance:
                if crit['singles'] == 1:
                    # We set the score to a large value to avoid
                    # being removed in reduction step
                    score.append((i, 1))
                    Chosen_ones.append(i)

                else:
                    if dif[i] > crit['gain_loss']:
                        score.append((i, dif[i] - crit['gain_loss']))
                        Chosen_ones.append(i)

            if abs(total_occ[i] - 2) < self.tolerance:
                if dif[i] > crit['lost_occ']:
                    score.append((i, dif[i] - crit['lost_occ']))
                    Chosen_ones.append(i)

        if self.dbug:
            print("")
            print("selected orbitals :", len(Chosen_ones), ",", np.array(Chosen_ones))
            print("selected electrons :", orbs_el_n(total_occ, Chosen_ones),
                  " DOMOs:", orbs_el_n(total_occ, Chosen_ones, cshell=True) / 2)
            print("")

        if len(Chosen_ones) > machine_limit:

            if self.dbug:
                print("Machine limit triggered")
                print("")

            Chosen_ones = self.reduce_orbs_n(Chosen_ones, nat_occ, score,
                                             machine_limit, max_loop=max_loop,
                                             incr_per=incr_per)

        # Check active space is correct
        if not AS_true(total_occ, Chosen_ones, tol=self.tolerance):
            raise Exception("Bad active space. electrons: ", orbs_el_n(total_occ, Chosen_ones),
                            "orbitals: ", Chosen_ones)

        if plot is True:
            value = {'occupied': crit['lost_occ'], 'virtual': crit['gained_occ']}
            if n_sing > 0:
                data = {'doubles': dif[:n_docc], 'singles': dif[n_docc:n_docc + n_sing],
                        'virtuals': dif[n_docc + n_sing:]}
            else:
                data = {'doubles': dif[:n_docc], 'virtuals': dif[n_docc + n_sing:]}
            plot_selection(data, value, 0, ptitle, xlabel='# MO', fontsize2=self.fontsize2,
                           fontsize=self.fontsize, linewidth=self.linewidth, fancybox=self.fancybox,
                           ylabel='deviation from SCF occupation', fname='MP2nat_occ_',
                           idxshift=self.idxshift)

        self.dif = dif
        self.crit = crit

        return list(Chosen_ones)

    def run_dmrg(self, mol: gto.mole.Mole, mo: np.ndarray, el: int, orbs_in: Union[int, list],
                 nroots: int = 1, symm: str = '', del_temp_: bool = False):
        r"""Function which performs a DMRG calculation using Block.
            Most of the flags are set within the init!
            Memory for Block is set as mol.max_memory unless defined by user.

            Args:
                mol:			            PySCF mol object.
                                            You can pass the mf here but we will only grab mol
                                            from it.
                mo (np.ndarray):            Orbital coefficients.
                el (int):                   Number of electrons.
                orbs_in (int/list):         Number or list of orbitals.
                nroots (int):               Number of roots to be calculated.
                symm (str):                 Symmetry of wavefunction.
                del_temp_ (bool):           Remove temporary files made by Block and interface.

            Returns:
                mc (object):                PySCF MCSCF object.
                t_taken (float):            Calculation time.

            raises:
                TypeError: look up the error.
                Exception: look up the error.
        """

        # BLOCK files:
        # mc.fcisolver.scratchDirectory
        # mc.fcisolver.runtimeDir

        # Set memory from mol if it not given
        if self.memory is None:
            self.memory = mol.max_memory / 1000
        # mf/mol clarification
        if not isinstance(mol, gto.mole.Mole):
            if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF, scf.rhf.RHF)):
                mol = mf.mol
            else:
                raise TypeError('First argument should be a mol or mf object')

        if not isinstance(mo, np.ndarray):
            raise TypeError('molecular orbital coefficients should be numpy array.')

        # Orbital clarification
        if not mo.ndim == 2:
            print("Warning! /n UHF orbitals passed grabbing alpha.")
            if mo.ndim == 3:
                mo = np.array(mo[0, :, :])
            else:
                raise TypeError('Orbitals should be one set RHF or two UHF',
                                'Orbital dimension: ', mo.ndim)

        if mol.symmetry is True and not symm:
            raise Exception("Assign symmetry of each root in symm as a list.")

        if symm and mol.symmetry is False:
            raise Exception("Symmetry assigned while mol object is not symmetric")
            # orbs_in is list of orbitals
        if isinstance(orbs_in, (np.ndarray, list)):
            orbs = len(orbs_in)
            # orbs_in is number of orbitals
        elif isinstance(orbs_in, int):
            orbs = orbs_in
        else:
            raise TypeError("Unknown orbital input type ", type(orbs_in))
        if self.dbug:
            print('-------DMRG-RUN-----------')
            print('el: ', el, ' orbs: ', orbs)
            print('-------------------------')
        mc = mcscf.CASCI(mol, orbs, el)
        if self.DMRG_MaxM is not None:
            mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=self.DMRG_MaxM, memory=self.memory)
        else:
            mc.fcisolver = dmrgscf.DMRGCI(mol, memory=self.memory)

        if mol.symmetry is True:
            mc.fcisolver.wfnsym = symm

        mc.fcisolver.spin = mol.spin

        # Set the DMRG configurations
        if len(self.DMRG_S) >= 1:
            mc.fcisolver.scheduleSweeps = self.DMRG_S
        if len(self.DMRG_M) >= 1:
            mc.fcisolver.scheduleMaxMs = self.DMRG_M
        if len(self.DMRG_tol) >= 1:
            mc.fcisolver.scheduleTols = self.DMRG_tol
        if len(self.DMRG_N) >= 1:
            mc.fcisolver.scheduleNoises = self.DMRG_N
        if self.tdod is not None:
            mc.fcisolver.twodot_to_onedot = self.tdod
        if self.DMRG_SM is not None:
            mc.fcisolver.startM = self.DMRG_SM

        mc.fcisolver.nroots = nroots

        if self.restart is True:
            mc.fcisolver.restart = self.restart

        # Watch about about this.
        # Let pyscf decide this
        # mc.fcisolver.maxIter = set_maxS

        if self.dbug:
            print("DMRG calculations on electrons", el, "orbitals", orbs)
            print("With M:", self.DMRG_M, " Sweeps:", self.DMRG_S, " MaxM:", self.DMRG_MaxM)
            print("")

        if self.nthreads is not None:
            mc.fcisolver.num_thrds = self.nthreads

        start_time = timeit.default_timer()
        # if orbs_in is a list of orbitals rotate them
        if isinstance(orbs_in, (np.ndarray, list)):
            mo = mcscf.sort_mo(mc, mo, np.array(orbs_in) + 1)
            mc.run(mo)
            # return rdm1AO as a list (multi-root support)
        else:
            mc.run(mo)

        self.t_taken = timeit.default_timer() - start_time

        if del_temp_ is True:
            print('del_temp is disabled')
            # del_DMRG_temp_(mc)

        return mc

    def dm(self, mc: Union[CASCIClass, CASSCFClass], mo: list = None,
           root: Union[int, list] = None) -> list:
        r"""Calculate orbdesn.

            Args:
                mc (CASCIClass):        PySCF mc object.
                mo (list):              Do not use this.
                                        Specified orbitals for rdm calculations.
                                        This is a list of orbitals for each root.
                                        If only one root the a one membered (np.ndarray) list.
                root (int/list):        Calculate for a specific root(int) or roots(list).
                                        Otherwise for all roots.
                                        In that case mo should be None or list of orbs for all
                                        roots.

            Returns:
                orbdens (list):         Orbital desnity required for entanglement
                                        calculations.

            raises:
                Exception: wrong inputs.
                ValueError: Look up errors.
                NotImplementedError: Do not use this feature in this version.
        """
        # Sanity Check
        if not self.tolerance >= 0.0:
            raise ValueError('tolerance should be positive.')
        # rroots is range of roots, for which we calculate the dm for
        if root is None:
            rroots = list(range(mc.fcisolver.nroots))
        elif isinstance(root, list):
            rroots = root
        elif isinstance(root, int):
            rroots = [root]
        else:
            raise Exception('Bad root entry')
        if mo is not None:
            for i in rroots:
                if not isinstance(mo[i], np.ndarray):
                    raise Exception('mo should be a numpy array')
        ncas = mc.ncas
        # Total number of roots
        nroots = mc.fcisolver.nroots
        # nmo = mc.mo_coeff.shape[1]
        # ncore = mc.ncore

        from pyscf.dmrgscf import DMRGCI

        orbdens = []
        # If only specific root/s is reuqested
        # You will get a list with dimensions of total
        # roots in MC with blanks for unrequested roots.
        for i in range(nroots):
            if i in rroots:
                if mo is None:
                    rdm1a, rdm1b = DMRGCI.make_rdm1s(mc.fcisolver, i, ncas, mc.nelecas)
                    rdm1, rdm2 = DMRGCI.make_rdm12(mc.fcisolver, i, ncas, mc.nelecas)
                elif isinstance(mo, list):
                    raise NotImplementedError('feature not developed yet')
                    # This somehow gives wrong results
                    rdm1, rdm2 = mcscf.addons.make_rdm12(mc, mo_coeff=mo[i], ci=i)
                    rdm1a, rdm1b = mcscf.make_rdm1s(mc, mo_coeff=mo[i], ci=i)

                # Just active
                if not (rdm1a.shape == (ncas, ncas)):
                    raise Exception('bad dimensions')
                if not (rdm1b.shape == (ncas, ncas)):
                    raise Exception('bad dimensions')
                if not (rdm2.shape == (ncas, ncas, ncas, ncas)):
                    raise Exception('bad dimensions')

                orbdens_i = np.zeros((mc.ncas, 4))

                for j in range(ncas):
                    rdm2s = 0.5 * rdm2[j, j, j, j]
                    orbdens_i[j, 0] = 1.0 - rdm1a[j, j] - rdm1b[j, j] + rdm2s
                    orbdens_i[j, 1] = rdm1a[j, j] - rdm2s
                    orbdens_i[j, 2] = rdm1b[j, j] - rdm2s
                    orbdens_i[j, 3] = rdm2s

                for i in range(ncas):
                    if not np.allclose(orbdens_i[i], orbdens_i[i].conjugate().T,
                                       atol=self.tolerance):
                        raise Exception('Density matrix should be hermitian')

                for j in range(ncas):
                    for k in [0, 1, 2, 3]:
                        if orbdens_i[j, k] < -self.tolerance:
                            raise Exception('warning negative value in orbdens.')

                if not ((orbdens_i.ndim == 2) and (orbdens_i.shape[1] == 4)):
                    raise Exception('bad dimensions')

                orbdens.append(orbdens_i)
            else:
                orbdens.append([])

        return orbdens

    def eval_ss_entropy(self, dm: np.ndarray) -> np.ndarray:
        r"""Function which evaluates the single site entropy

            Args:
                dm (np.array):          Density matrix of active orbitals.

            Returns:
                s (np.array):           The single site entropy.

            Raises:
                ValueError: look up errors
        """
        if not self.tolerance >= 0.0:
            raise ValueError('tolerance should be positive.')
        s = []

        # Locations where the one-orbital density is positive
        where_positive = (dm > self.tolerance)
        # Locations where the one-orbital density is zero (within the self.tolerance)
        where_zero = np.logical_and(dm <= self.tolerance, dm >= -self.tolerance)
        # Ensure there are no negative entries
        if not np.all(dm >= -self.tolerance):
            raise ValueError('Negative values in density matrix')

        WlogW = dm * np.log(dm, where=where_positive)
        WlogW[where_zero] = 0.0
        s = -np.sum(WlogW, axis=1)

        for i in s:
            if i < -self.tolerance:
                print('Warning negative entropy!')

        return s

    def eval_ss_entropy_wrapper(self, mc: Union[CASCIClass, CASSCFClass], mo: np.ndarray = None,
                                root: int = None) -> np.ndarray:
        r"""Wrapper for entropy calculation functions.

            Args:
                mc (CASCIClass):        PySCF mc object.
                mo (array):             Do not use this feature.
                                        Specific orbitals for rdm calculations.
                root (int):             Calculate for a specific root(int) or roots(list).
                                        Otherwise for all roots.

            Returns:
                s (np.array):           The single site entropy for the given MO.
        """

        nroots = mc.fcisolver.nroots
        s = []
        if root is None:
            rroots = list(range(mc.fcisolver.nroots))
        elif isinstance(root, int):
            rroots = [root]
        elif isinstance(root, list):
            rroots = root
        else:
            print('Bad root entry')

        orbdens = self.dm(mc, mo=mo, root=root)

        for i in range(nroots):
            if i in rroots:
                s_i = self.eval_ss_entropy(orbdens[i])
                s.append(s_i)
            else:
                s.append([])

        return s

    def threshold_set(self, entropies: list, Th_auto: str = 'Max',
                      threshold: float = 0.13) -> list:
        r"""Calculation of entropy selection threshold.

            Args:
                entropies (list):           List of entropies per root.
                                            [[entropies of root1], [entropies of root2]]
                Th_auto (str):              Automatic derivation of thresholds.
                                            'HQS' default.
                                            'STD' stanorredard deviation is used.
                                            'AVG' average value is used.
                                            'Max' selects the max value.
                                            'None' uses the Threshold value (default 0.13).
                                            Case insensitive.
                threshold(float):           Threshold single site entropy which defines the border
                                            between active and inactive MOs.
                                            The standard value is 0.13.
                                            There is trigger which redefines the threshold.
                                            Then Thresholds will be defined by the software
                                            automatically. You can Change this by setting Th_auto to
                                            'None'.

            Returns:
                Th_list (list):             List of thresholds selected per root.

            self:
                p_thresh(list of dict):     All the calculated thresholds.

            raises:
                Exception: Th_auto not recognized.
        """
        Th_list = []
        self.p_thresh = []
        # Calculate thresholds for each root
        for i in range(len(entropies)):
            # set the smallest entropy to 0
            dif = entropies[i] - np.min(entropies[i])
            # debugging
            if self.dbug:
                print("")
                print("Threshold selection info for root #", i)
                print("HQS:", sum((dif - np.mean(dif)**2) / (len(dif) - 1)),
                      "avg:", np.average(dif), "STD:", np.std(dif))
           if max(entropies[i]) < 0.13:
                # Warn anyway
                print('mas:', max(entropies[i]))
                print("Warning! System is most probably single-reference")
            # Calculate all possible thresholds
            self.p_thresh.append({})
            self.p_thresh[i]['AVG'] = np.average(dif)
            self.p_thresh[i]['HQS'] = sum((dif - np.mean(dif)**2) / (len(dif) - 1))
            self.p_thresh[i]['STD'] = np.std(dif)

            # The trigger to redefination of entropy
            # If the Automatically calculated thresholds are too small
            # We grab the minimum value of 0.13 or user defined
            if threshold < self.p_thresh[i]['AVG'] or \
               threshold < self.p_thresh[i]['STD'] or threshold < self.p_thresh[i]['HQS']:

                if Th_auto.upper() == 'HQS':
                    if threshold < self.p_thresh[i]['HQS']:
                        value = self.p_thresh[i]['HQS']
                        # For Plotting, purpose.
                        # If selected: 1. convert to a list add user set color to it.
                        self.p_thresh[i]['HQS'] = [self.p_thresh[i]['HQS']]
                        self.p_thresh[i]['HQS'].append(self.dcolor)
                    else:
                        value = threshold
                        self.p_thresh[i]['other'] = [value, self.dcolor]

                elif Th_auto.upper() == 'STD':
                    if threshold < self.p_thresh[i]['STD']:
                        value = self.p_thresh[i]['STD']
                        self.p_thresh[i]['STD'] = [self.p_thresh[i]['STD']]
                        self.p_thresh[i]['STD'].append(self.dcolor)
                    else:
                        value = threshold
                        self.p_thresh[i]['other'] = [value, self.dcolor]

                elif Th_auto.upper() == 'AVG':
                    if threshold < self.p_thresh[i]['STD']:
                        value = self.p_thresh[i]['AVG']
                        self.p_thresh[i]['AVG'] = [self.p_thresh[i]['AVG']]
                        self.p_thresh[i]['AVG'].append(self.dcolor)
                    else:
                        value = threshold
                        self.p_thresh[i]['other'] = [value, self.dcolor]

                elif Th_auto.upper() == 'MAX':
                    value = max(self.p_thresh[i]['AVG'], self.p_thresh[i]['HQS'],
                                self.p_thresh[i]['STD'])

                    # Find the chosen one add color tag to it
                    for key, kvalue in self.p_thresh[i].items():
                        if value - kvalue <= self.tolerance:
                            self.p_thresh[i][key] = [self.p_thresh[i][key], self.dcolor]
                # Do not change the treshold.
                elif Th_auto == 'None':
                    value = threshold
                    self.p_thresh[i]['other'] = [value, self.dcolor]
                else:
                    raise Exception("Th_auto value not recognized")

                if self.dbug:
                    print("Threshold set to", value, 'for root #', i)
                    print("")

                Th_list.append(value)
            else:
                if self.dbug:
                    print("Threshold set to", threshold, 'for root #', i)
                Th_list.append(threshold)

        return Th_list

    def select_mos_ent(self, entropies: list, Th_list: list, occup: Union[list, np.ndarray],
                      first_active: int, test: bool = True, singles: bool = True,
                      homo_lumo: bool = True, Th_reduce: float = 0.01,
                      qfalse: bool = True):
        r"""Select MOs based on the threshold and user input.

            Args:
                entropies (list):           List of entropies per root.
                                            [[entropies of root1], [entropies of root2]]
                Th_list (list):             List of selected thresholds.
                occup (list):               Orbital occupation.
                first_active (int):         index of first active orbital.
                test (bool):                If set to True, I will check for correction
                                            of selected active space.
                singles (bool):             Singly occupied orbitals are considered active.
                                            Change at your own peril.
                homo_lumo (bool):           Add humo and lomo orbitals to the selection.
                                            The selection is only for occup value you enter.
                                            Only for groundstate. (As we don't have excited
                                            state occupation. This will be added in future
                                            releases.)
                Th_reduce (float):          Howmuch to reduce threshold by until correct
                                            active space is found.
                qfalse (bool):              Not quick after active space selection failure.

            Returns:
                act_mos (list):             Active orbitals.
                act_ele (list):             Active electrons.
                Th_list (list):             List of selected thresholds.

            self:
                p_thresh (list of dict):     All the calculated thresholds.

            raises:
                Exception: Selected active space is not rational.
        """

        act_mos = []
        act_ele = []

        # Loop through all roots
        for i in range(len(entropies)):
            ASC = False
            occ_flag = True  # The first orbitals are supposed to be occupied
            # Loop over until meaningful active space is given
            while ASC is False:
                act_mos_i = []
                act_ele_i = int(0)
                for ii, entropy in enumerate(entropies[i]):
                    # singly occupied orbitals
                    if singles is True and abs(occup[ii + first_active] - 1) < self.tolerance:
                        act_mos_i.append(ii + first_active)
                    # HOMO LUMO only for i==0 first root.
                    elif homo_lumo is True and occ_flag is True and \
                        abs(occup[ii + first_active]) < self.tolerance and i == 0:
                        # No more occupied orbitals
                        occ_flag = False
                        HUMO = ii + first_active
                        # Add LOMO if it is not included
                        if HUMO - 1 not in act_mos_i:
                            act_mos_i.append(HUMO - 1)
                        # Add the HUMO
                        act_mos_i.append(HUMO)
                    else:
                        # Check entropy value
                        if entropy >= Th_list[i]:
                            act_mos_i.append(ii + first_active)
                    # If Check if the orbital is occupied
                    # For HOMO LUMO selection
                    if abs(occup[ii + first_active]) > self.tolerance:
                        occ_flag = True

                act_ele_i = orbs_el_n(occup, act_mos_i)

                if AS_true(occup, act_mos_i, tol=self.tolerance):
                    ASC = True
                elif test is False:
                    ASC = True
                else:
                    Th_list[i] = Th_list[i] - Th_reduce
                    if self.dbug:
                        print("Calculations failed", Th_list[i] + Th_reduce, 'for root #', i,
                              'reduced to', Th_list[i])
                        print("Active electrons", act_ele_i, 'orbitals', act_mos_i)

                # Fake true flag to quit calculations
                if Th_list[i] < 0.02:
                    ASC = True

            # Orbital numbers is shifted already
            # Append each roots results
            act_mos_i.sort()
            act_mos.append(act_mos_i)
            act_ele.append(act_ele_i)

            # If still active space selection is not useful CRASH
            if not AS_true(occup, act_mos_i, tol=self.tolerance):
                if qfalse is True:
                    raise Exception('Bad active space selected...')
                else:
                    print('Warning bad active space!')

        return act_mos, act_ele, Th_list

    def find_active_space(self, mol: gto.mole.Mole, el: int, orbs: Union[int, list],
                          mo: np.ndarray, occup: list = None, nroots: int = 1,
                          singles: bool = True, threshold: float = 0.13, plot: bool = False,
                          vis_mos: bool = False, symm: str = '', save_entropy: bool = False,
                          save_occ: bool = False, Th_auto: str = 'HQS', Dsym: bool = False,
                          qfalse: bool = False, Th_reduce: float = 0.01, del_temp_: bool = False,
                          entropies: list = None, homo_lumo: bool = True):
        r"""Main function of this module which performs an active space guess
            based on a quick and dirty 1/2RDM calculation via DMRGCI.

            Args:
                mol (object):    	        An instance of pyscf mol class.
                el (int):                   Number of electrons. Use orbs_el_n to find number of
                                            elecrons in a list.
                orbs (int/list): 	        Is the active space of the large DMRGCI calculation
                                            Either int or [list of orbitals].
                                            I expect counting from 0.
                mo (np.ndarray):            Molecular orbital coefficients.
                occup (list): 	            Orbital occupation.
                                            Required for singles option if not given a fake
                                            occupation is made using mol's spin.
                nroots (int):               The number of roots to be calculated.
                                            We do two sepaerate calculattions.
                                            This will return lists per root.
                                            Two active spaces, two natural orbitals etc.
                singles (bool):             Singly occupied orbitals are considered active.
                                            Change at your own peril.
                homo_lumo (bool):           Add humo and lomo orbitals to the selection.
                                            The selection is only for occup value you enter.
                                            Only for groundstate. (As we don't have excited
                                            state occupation. This will change in next realease.)
                threshold(float):           Threshold single site entropy which defines the border
                                            between active and inactive MOs.
                                            The standard value is 0.13.
                                            There is a trigger which redefines the threshold.
                                            Then Thresholds will be defined by the software
                                            automatically. You can Change this by setting Th_auto to
                                            'None'.
                plot (bool):                If set to 'True' the calculated MO single site entropies
                                            will be plotted ('plot_selection' will be executed).
                vis_mos (bool):             If set to 'True' the active MOs will be plotted to PNG
                                            files via Jmol. For this, Jmol has to be installed on
                                            your System and in your $PATH!
                symm (str):                 Symmetry of the wavefunction. ['A1','B1] for 2 roots.
                                            Must be provided if symmetry is used!
                save_entropy(bool):         Save the entropies a csv file (entropies).
                                            File name can be changed by setting filename_ent in
                                            init.
                save_occ (bool):            If set to True, a .csv file ('nat_occ') with
                                            File name can be changed by setting filename_nat in
                                            init.
                                            Set to a string if you want to specify the file name.
                Th_auto (str):              Automatic derivation  of thresholds.
                                            'HQS' default.
                                            'STD' stanorredard deviation is used.
                                            'AVG' average value is used (recommended for closed
                                            shell systems).
                                            'Max' selects the max value.
                                            'None' uses the Threshold value (default 0.13).
                                            Case insensitive.
                Dsym (bool):                Force disable symmetry.
                qfalse (bool):              Not to quit after active space selection failure.
                Th_reduce (float):          Howmuch to reduce threshold by until correct
                                            active space is found.
                del_temp_ (bool):           Remove temporary files made by Block and interface.
                entropies (list):           Manually enter entropies will skip DMRG calculations.
                                            Same as self.entropies.

            Returns:
                act_ele (int):	            The suggested active electrons.
                act_mos (list):	            The suggested active MOs as a list. Counting from 0.

            self:
                mc (object):                PySCF MC object.
                no (list):                  List of natural orbitals for each root.
                occ (list):                 List of natural orbital occupations for each root.
                mo_entropies (list):        MO entropies.

            raises:
                ValueError: Sum of natural orbital occupations does not match total number of
                            electrons.
                TypeError: look up the error.
                Exception: look up the error.

            Note:
                All MO indices are given back as 0-based list.
        """

        # Sanity check
        if not isinstance(mol, gto.mole.Mole):
            raise TypeError('mol should be an instance of PyCSF mol class')
        if not isinstance(mo, np.ndarray):
            raise TypeError('mo coefficients must be a numpy array')
        if mo.ndim != 2:
            raise Exception('RHF or orbitals of similar dimensions should be inputed')
        if not isinstance(el, (int, np.int64)):
            raise TypeError('number of enelectrons should be an int')
        if not isinstance(orbs, (int, list, np.ndarray)):
            raise TypeError('orbs should be a list of orbital indexes or an int',
                            'for number of orbitals')
        if isinstance(orbs, np.ndarray):
            orbs = list(orbs)

        if not isinstance(nroots, int):
            raise TypeError('nroots most be an integer')

        if occup is not None:
            if not isinstance(occup, (list, np.ndarray)):
                raise TypeError('Occupation must be entered as list or numpy array')
            if len(occup) == 2:
                raise Exception('Provide one dimensional occupation (RHF)')
        else:
            occup = scf_mo_occ(mol)


        if symm:
            if not isinstance(symm, str):
                raise TypeError('Symmetruy most be entered as a string')

        if mol.symmetry is True:
            if not symm:
                raise Exception("Assign symmetry of each root in symm as a string.")

        # Let's keep list as lists
        if isinstance(orbs, np.ndarray):
            orbs = list(orbs)

        # If entropies are given Skip DMRG calculations.
        if entropies is not None:
            if not isinstance(entropies, (list, np.ndarray)):
                raise TypeError('entropies most be a list or numpy array.')
            if isinstance(entropies, np.ndarray):
                entropies = list(entropies)
            self.entropies = entropies

        if len(self.entropies) == 0:
            self.run_dmrg_flag = True
        else:
            # If the active space size is different than entropies, do DMRG.
            if isinstance(orbs, list):
                if len(self.entropies[0]) != len(orbs):
                    self.run_dmrg_flag = True
            elif isinstance(orbs, int):
                if len(self.entropies[0]) != orbs:
                    self.run_dmrg_flag = True
            else:
                self.run_dmrg_flag = False

        # Echo to user
        if self.verbose != 0:
            print("-----------ASF-RUN----------")
            print("Active space: ", orbs)
            print("Active electrons: ", el)
            print("Entropies in memory or entered manually now: ", self.entropies)
            print("Run DMRG: ", self.run_dmrg_flag)
            print("symmetry: ", symm)
            print("Spin, Multiplicity: ", mol.spin, ',', mol.spin + 1)
            print("")

        # If We don't resume do DMRG
        if self.run_dmrg_flag is True:
            mc = self.run_dmrg(mol, mo, el, orbs, nroots=nroots, symm=symm, del_temp_=del_temp_)

            # Set settings
            ncas = mc.ncas

            # I believe this is alpha and beta electrons
            nele = sum(mc.nelecas)
            first_active = act_fao(occup, nele)

            no = []
            occ = []
            # calculate the natural orbitals
            # Solve eigenvalue problem for each root
            rdm1AO = []
            mo_entropies = []

            for i in range(nroots):
                # Calculate the entropies
                mo_entropies.append(self.eval_ss_entropy_wrapper(mc, root=i)[i])

                if self.dbug:
                    print("Natural orbitals orbitals for root:", i)

                # RDM1 AO basis
                rdm1AO = mcscf.make_rdm1(mc, ci=i)

                # Calculate the natural orbitals
                no_i, occ_i = nat_orbs_rdm1AO_rhf(rdm1AO, mc.mo_coeff, mol)

                if not sum(occ_i) - mol.nelectron < self.tolerance:
                    raise ValueError('Natural orbital occupation does not reflect',
                                     'Total number of electrons')
                if self.dbug:
                    print("Sum of natural occupation", sum(occ_i))
                    print("")

                no.append(no_i)
                occ.append(occ_i)

            # Set the values aceessible
            self.entropies = mo_entropies
            self.mc = mc
            self.no = no
            self.occ = occ

            # Save if asked for
            if save_occ is not False:
                if not self.filename_nat:
                    self.filename_nat = "occupation_M_" + str(self.DMRG_M) + "_S_" + \
                        str(self.DMRG_S) + ".csv"
                else:
                    save_csv(self.filename_nat, occ)

        elif isinstance(self.entropies, list):
            mo_entropies = self.entropies
            first_active = act_fao(occup, el)
            occ, no = [], []

        # Save entropies if requested
        if save_entropy is not False:
            # Auto assigned name.
            if not self.filename_ent:
                self.filename_ent = "entropis_M_" + str(self.DMRG_M) + "_S_" + str(self.DMRG_S) \
                    + ".csv"
            save_csv(self.filename_ent, mo_entropies)

        # Select entropy thresholds
        Th_list = self.threshold_set(mo_entropies, Th_auto=Th_auto, threshold=threshold)

        # Select the orbitals
        act_mos, act_ele, Th_list = self.select_mos_ent(mo_entropies, Th_list, occup,
                                                       singles=singles, homo_lumo=homo_lumo,
                                                       Th_reduce=Th_reduce,
                                                       first_active=first_active, qfalse=qfalse)

        # Plot entropies if requested
        if plot is True:
            for i in range(nroots):
                plot_selection(mo_entropies[i], self.p_thresh[i], i, idxshift=first_active,
                               fontsize2=self.fontsize2, fontsize=self.fontsize,
                               linewidth=self.linewidth, fancybox=self.fancybox)

        # Visualize the orbitals if requested
        if vis_mos is True:
            for i in range(nroots):
                visualize_mos(mol, no[i], act_mos[i], i)

        # logger
        if self.verbose != 0:
            if self.run_dmrg_flag is True:
                print('DMRG calculation was performed on %s e- in %s MOs' % (nele, ncas),
                      '(w/ %s DMRG M, %s DMRG Sweeps,' % (self.DMRG_M, self.DMRG_S),
                      ' %s DMRG_MaxM)' % (self.DMRG_MaxM))
                print('Time taken for DMRG calculation: %s seconds'
                      % self.t_taken)

                print('\nSingle site MO entropies and orbital occupations:')
                print('Starting from 0.')
                for i in range(nroots):
                    print('root # %s' % (i))
                    for ii, entropy in enumerate(mo_entropies[i]):
                        print('S[MO %s] = %1.4f, ' % (ii + first_active, entropy),
                              'natural occ = %1.4f, ' % (occ[i][ii + first_active]),
                              'SCF OCC = %1.4f' % (occup[ii + first_active]))
            else:
                if self.verbose <= 3:
                    print('\nSingle site MO entropies and orbital occupations:')
                    print('Starting from 0.')
                    for i in range(nroots):
                        print('root # %s' % (i))
                        for ii, entropy in enumerate(mo_entropies[i]):
                            print('S[MO %s] = %1.4f, ' % (ii + first_active, entropy),
                                  'SCF OCC = %1.4f' % (occup[ii + first_active]))
        if self.verbose != 0:
            for i in range(nroots):
                print("\nASF suggested active space for root #%s:\
                      \nPut %s e- in these %s MOs:%s with TH:%s\n"
                      % (i, act_ele[i], len(act_mos[i]), act_mos[i], Th_list[i]))

        return act_ele, act_mos

    def fas_no_guess(self, mf: Union[scf.uhf.UHF, scf.rohf.ROHF, scf.rhf.RHF],
                     nat_type: str = 'MP2', machine_limit: int = 30, symm: str = '',
                     nroots: int = 1, Th_auto_nat: str = 'HQS', Th_auto_DMRG: str = 'HQS',
                     converge_: bool = False, add_orb_id: list = None, Ms: list = None,
                     MaxM: int = 2000, init_M: int = 31, init_S: int = 0, Max_iter: int = 24,
                     ent_conv: float = 0.005, ent_conv_max: float = 0.0, S_inc: int = 1,
                     spin: int = -1, plot: bool = False):
        r"""A simple wrapper for active space search.

            Args:
                mf (object):                A converged meanfield object.
                nat_type (str):             Type of the natural orbitals.
                                            'MP2', 'CISD' and 'CCSD'.
                machine_limit (int):        Maximum allowed selection.
                symm (str):                 Symmetry of the wavefunction. ['A1'. 'B1'] for 2 roots.
                                            Must be provided if symmetry is used!
                nroots (int):               The number of roots to be calculated.
                                            We do two sepaerate calculattions.
                                            This will return lists per root.
                                            Two active spaces, two natural orbitals etc.
                Th_auto_nat (str):          Type of automatic threshold selection for
                                            natural orbitals.
                Th_auto_DMRG (str):         Type of automatic threshold selection for
                                            DMRG entropies.
                converge_ (bool):           This function is for development purpose.
                                            This feature is not rigorously tested.
                add_orb_id (list):          Set if you want a set of specific orbitals
                                            To be included in DMRG calculations.
                                            [4,5]
                Ms (list):                  List of M values for DMRG.
                                            Set the StartM with DMRG_SM if necessary.
                MaxM (int):                 Will change the MaxM when using the converge function.
                init_M (int):               Starting M value, when using the converge function.
                init_S (int):               Starting S value, when using the converge function.
                Max_iter (int):             Maximum number of iterations, when using converge.
                ent_conv (float):           Convergence threshold (avergae of entropies).
                ent_conv_max (float):       Max deviation from last step.
                                            Default is 0 which means 5 x ent_conv.
                S_inc (int):                Increment for increasing sweeps.
                spin (int):                 If set to a positive value DMRG calculations
                                            will be carried out in that multipilicity.
                plot (bool):                Threshold selection plot for both natural orbitals
                                            and entropy.

            Returns:
                act_ele (int):	            The suggested active electrons.
                act_mos (list):	            The suggested active MOs as a list. Counting from 0.

            self:
                init_nat (np.ndarray):      Initial CISD, MP2, etc nat orbs
                initi_occ (np.ndarray):     Initial guess occupations.
                AVG (list):                 AVG convergence.
                ent (list):                 ent during convergence.

            raises:
                TypeError:                  Read the errors.

        """
        print('--------Wrapper function--------')
        print('')
        if not isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF, scf.rhf.RHF)):
            raise TypeError('mf object is wrong')

        from pyscf import mp, ci, cc

        mol = mf.mol

        # Set default ent_conv_max value
        if abs(ent_conv_max - 0) < self.tolerance:
            ent_conv_max = 5 * ent_conv
        else:
            if not isinstance(ent_conv_max, (int, float, np.float)):
                raise TypeError('ent_conv_max should be a float')

        if isinstance(mf, scf.uhf.UHF):
            occu = mf.mo_occ[0, :] + mf.mo_occ[1, :]
        else:
            occu = mf.mo_occ

        if nat_type == 'MP2':
            obj = mp.MP2(mf).run()
        elif nat_type == 'CISD':
            obj = ci.CISD(mf).run()
        elif nat_type == 'CCSD':
            obj = cc.CCSD(mf).run()
        else:
            raise TypeError('Type of natural orbitals not understood')

        no, occ = nat_orbs_obj(obj)
        self.init_nat = no
        self.initi_occ = occ

        sel_mos = self.select_mos_occ(occ, Th_auto=Th_auto_nat, machine_limit=machine_limit,
                                      plot=plot)
        print('---------------------')
        print('Natural orbital selected active space')
        print('orbitals:', len(sel_mos), ', electrons:', orbs_el_n(mf.mo_occ, sel_mos))
        print('---------------------')
        print('')
        if add_orb_id is not None:
            sel_mos = list(set(sel_mos) | set(add_orb_id))

        if spin >= 0:
            mol.spin = spin
            mol.build()

        if converge_:
            self.DMRG_MaxM = MaxM
            ent: List[List[List[float]]]
            ent = []
            AVG = []
            if Ms is None:
                Ms = []
                for i in range(Max_iter):
                    Ms.append(i * 25 + init_M)

                if self.DMRG_SM is not None:
                    if self.DMRG_SM > init_M:
                        print('Warning init_M smaller than DMRG_SM')
                        print('Changing DMRG_SM to init_M - 5')
                        self.DMRG_SM = init_M - 5

            for i, m in enumerate(Ms):
                self.DMRG_M = [m]

                if i == 0:
                    self.DMRG_S = [init_S]
                else:
                    self.DMRG_S = [self.DMRG_S[0] + S_inc]
                if i < 6:
                    self.DMRG_tol = [1.0000e-07]
                else:
                    self.DMRG_tol = [5.0000e-10]
                if i < 5:
                    self.DMRG_N = [5.0000e-05]
                else:
                    self.DMRG_N = [0.0]

                print('#####################')
                print('Run ', i, ', M', self.DMRG_M)
                print('#####################')

                act_ele, act_mos = self.find_active_space(mol, orbs_el_n(mf.mo_occ, sel_mos),
                                                          sel_mos, no, occup=occu,
                                                          symm=symm, nroots=nroots,
                                                          Th_auto=Th_auto_DMRG)

                ent.append(self.entropies)

                if len(ent) >= 2:
                    AVG_i = []
                    for i in range(len(ent[-1])):
                        max_dev = max(abs(np.array(ent[-1][i]) - np.array(ent[-2][i])))
                        AVG_i.append(np.average(abs(np.array(ent[-1][i]) - np.array(ent[-2][i]))))

                    AVG.append(np.average(AVG_i))

                    if AVG[-1] < ent_conv:
                        if max_dev < ent_conv_max:
                            print('###########################################')
                            print('DMRG converged at M and S:', self.DMRG_M, self.DMRG_S)
                            print('AVG and Max deviation:', AVG[-1], max_dev)
                            print('###########################################')
                            print('')
                            break
                    else:
                        print('##############################################')
                        print('AVG dev.', AVG[-1], 'Max dev.', max_dev, 'M', self.DMRG_M, 'S',
                              self.DMRG_S)
                        print('##############################################')

                self.entropies = []
                self.restart = True
        else:
            act_ele, act_mos = self.find_active_space(mol, orbs_el_n(mf.mo_occ, sel_mos), sel_mos,
                                                      no, occup=occu, symm=symm, nroots=nroots,
                                                      Th_auto=Th_auto_DMRG)

        if converge_:
            self.ent = ent
            self.AVG = AVG

        if plot is True:
            first_active = act_fao(mf.mo_occ, act_ele)
            for i in range(nroots):
                plot_selection(self.entropies[i], self.p_thresh[i], i, idxshift=first_active,
                    fontsize2=self.fontsize2, fontsize=self.fontsize,
                    linewidth=self.linewidth, fancybox=self.fancybox)

        return act_ele, act_mos


if __name__ == "__main__":
    ASF = asf()
    mos = [[3, 5, 7, 8, 9], [3, 4, 7, 8]]
    mos_0 = ASF.select_mroot_AS(mos)

    mos_1 = np.array(ASF.select_mroot_AS(mos, base=1))
    assert np.array_equal(mos_0, [3, 4, 5, 7, 8, 9])
    assert np.array_equal(mos_0, mos_1 - 1)


    from pyscf import gto, ci
    mol = gto.Mole()
    mol.atom = """
    C          0.00000        0.00000        0.00000
    C          0.00000        0.00000        1.20000
    """
    mol.basis = 'def2-svp'
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 2
    mol.max_memory = 4000
    mol.symmetry = True
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    ASF.verbose = 5
    ASF.dbug = True
    # Remove old log
    if os.path.isfile('asf.log'):
        os.remove('asf.log')

    # Set DMRG to cheap
    ASF.DMRG_MaxM = 25
    ASF.nthreads = 8

    # Selected guess
    ele, mos = ASF.find_active_space(mol, orbs_el_n(mf.mo_occ, [3, 4, 5, 6, 7]),
                                     [3, 4, 5, 6, 7], mf.mo_coeff,
                                     symm='A1g',
                                     save_entropy=False)

    # Test DMRG natural orbital occupation.
    # DMRG no is list of lists for roots!
    sel_mos = ASF.select_mos_occ(ASF.occ[0], clipping=False)

    ASF.entropies = []
    # electrons in orbitals
    ele, mos = ASF.find_active_space(mol, 4, 6, mf.mo_coeff,
                                     symm='A1g',
                                     del_temp_=True,
                                     save_entropy=False)

    # Let's try CISD natorbs
    mci = ci.UCISD(mf)
    mci.kernel()

    # Calculate natural orbitals
    no, occ = nat_orbs_obj(mci)

    # Select Natural orbitals
    sel_mos = ASF.select_mos_occ(occ, plot=True, machine_limit=6, Th_auto='Min', max_loop=2)

    ASF.entropies = []
    ele, mos = ASF.find_active_space(mol, orbs_el_n(mf.mo_occ, sel_mos),
                                     sel_mos, no,
                                     symm='A1g',
                                     del_temp_=True,
                                     save_entropy=False)
    # No guess
    ASF.entropies = []
    ele, mos = ASF.fas_no_guess(mf, symm='A1g')
