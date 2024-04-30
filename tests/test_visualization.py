# Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.axes import Axes

from asf import ASFCI
from asf.rimp2_pairinfo import diag_cumulant_ump2
from asf.visualization import (
    draw_diagonal_cumulant,
    draw_overlap_active,
    draw_overlap_with_unrestricted,
    draw_pair_information,
    draw_pair_information_uhf,
)

np.random.seed(42)


def get_node_number(ax: Axes) -> int:
    """Determine number of nodes from Axes object."""
    n_nodes = 0
    try:
        nodes = next(c for c in ax.collections if isinstance(c, PathCollection))
        n_nodes = len(nodes.get_offsets())
    except StopIteration:
        print("Cannot determine number of nodes from PathCollection.")
    return n_nodes


def get_edge_number(ax: Axes) -> int:
    """Determine number of edges from Axes object."""
    n_edges = 0
    try:
        edges = next(c for c in ax.collections if isinstance(c, LineCollection))
        n_edges = len(edges.get_linewidths())
    except StopIteration:
        print("Cannot determine number of edges from LineCollection.")
    return n_edges


def test_draw_overlap_active():
    ovlp = np.random.rand(4, 4)
    _, ax = plt.subplots()
    draw_overlap_active(ovlp, ax=ax, check_overlap=False)

    n_nodes = get_node_number(ax)
    n_edges = len([p for p in ax.patches if isinstance(p, FancyArrowPatch)])
    assert n_nodes >= ovlp.shape[0]
    assert n_edges >= 0


def test_draw_overlap_with_unrestricted(formaldehyde_RHF, formaldehyde_UHF):
    mol = formaldehyde_RHF.mol
    mo_coeff_rhf = formaldehyde_RHF.mo_coeff
    mo_coeff_uhf = formaldehyde_UHF.mo_coeff
    ovlp_ao = mol.intor_symmetric("int1e_ovlp")
    ovlp_a = mo_coeff_rhf.T @ ovlp_ao @ mo_coeff_uhf[0]
    ovlp_b = mo_coeff_rhf.T @ ovlp_ao @ mo_coeff_uhf[1]
    _, ax = plt.subplots()
    draw_overlap_with_unrestricted((ovlp_a, ovlp_b), ax=ax)

    n_nodes = get_node_number(ax)
    n_edges = len([p for p in ax.patches if isinstance(p, FancyArrowPatch)])
    assert n_nodes >= mo_coeff_uhf.shape[1] + mo_coeff_uhf.shape[2]
    assert n_edges >= 0


def test_draw_pair_information():
    # partial pair information from ethene example (already negated)
    pair_info = np.array(
        [
            [-0.00440, -0.00286, -0.00318, -0.00251, -0.00257, 0.00535],
            [-0.00286, -0.00521, -0.00511, -0.00278, -0.00205, 0.00218],
            [-0.00318, -0.00511, -0.00682, -0.00266, -0.00029, 0.00041],
            [-0.00251, -0.00278, -0.00266, -0.00717, -0.00826, 0.00848],
            [-0.00257, -0.00205, -0.00029, -0.00826, -0.05023, 0.11171],
            [0.00535, 0.00218, 0.00041, 0.00848, 0.11171, -0.05103],
        ]
    )
    _, ax = plt.subplots()
    draw_pair_information(pair_info, ax=ax)

    n_nodes = get_node_number(ax)
    n_edges = get_edge_number(ax)
    assert n_nodes == pair_info.shape[1]
    assert n_edges == n_nodes * (n_nodes + 1) // 2


def test_draw_diagonal_cumulant(formaldehyde_RHF):
    sf = ASFCI(formaldehyde_RHF.mol, formaldehyde_RHF.mo_coeff, nel=4, mo_list=[6, 7, 8, 9])
    sf.calculate()
    rdm1s = sf.rdm1s()
    rdm2 = sf.rdm2()

    _, ax = plt.subplots()
    draw_diagonal_cumulant(rdm1s[0], rdm1s[1], rdm2, ax=ax)

    n_nodes = get_node_number(ax)
    n_edges = get_edge_number(ax)
    assert n_nodes == len(sf.mo_list)
    assert n_edges == n_nodes * (n_nodes + 1) // 2


def test_draw_pair_information_uhf(OH_radical_DFUMP2):
    mo_coeff = OH_radical_DFUMP2._scf.mo_coeff  # no public getter for SCF object
    pair_info = -diag_cumulant_ump2(OH_radical_DFUMP2, mo_coeff)

    _, ax = plt.subplots()
    draw_pair_information_uhf(pair_info)

    n_nodes = get_node_number(ax)
    n_edges = get_edge_number(ax)
    assert n_nodes == pair_info.shape[1] + pair_info.shape[2]
    assert n_edges == n_nodes * (n_nodes + 1) // 2
