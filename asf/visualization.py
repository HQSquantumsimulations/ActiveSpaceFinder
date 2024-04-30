# Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.

"""Analyze and draw data as networks."""

from typing import Optional, Union

import networkx
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes

from .pairinfo import cumulant2b
from .utility import calculate_entropy_s1, orbdens_from_rdm


def draw_overlap_active(
    S: np.ndarray,
    mo_lists: Optional[tuple[Union[list[int], np.ndarray], Union[list[int], np.ndarray]]] = None,
    labels: Optional[tuple[Union[list, np.ndarray], Union[list, np.ndarray]]] = None,
    lower_thresh: float = 0.15,
    upper_thresh: float = 0.85,
    active_colors: tuple[str, str] = ("royalblue", "crimson"),
    inact_colors: tuple[str, str] = ("forestgreen", "gold"),
    edge_color: str = "black",
    edge_width: float = 5.0,
    directed: bool = True,
    seed: int = 0,
    check_overlap: bool = True,
    ax: Optional[Axes] = None,
) -> None:
    """Set up the overlap of orbitals as a network, and draw it.

    This function is intended to visualize if two active spaces are consistent, or whether
    components of MOs outside of the active space have been picked up.

    Args:
        S:              Overlap matrix in MO basis.
        mo_lists:       Lists of indices of MO coefficients in the active space.
                        The first element of the tuple contains the list of MOs in the first active
                        space (rows of S). The second element of the tuple contains the list of MOs
                        in the second active space (columns of S).
        labels:         The labels for both sets of orbitals.
                        By default, orbitals are labelled as integers starting from 0.
        lower_thresh:   Do not draw a graph edge of the square of the overlap is below this cutoff.
        upper_thresh:   Omit orbitals (nodes) if the largest square of an overlap involving this
                        orbital with another active partner orbital is above this threshold.
        active_colors:  Colors to use for active orbitals.
        inact_colors:   Colors to use for inactive orbitals.
        edge_color:     Color used to draw the edges.
        edge_width:     Maximum width to draw the edges.
        directed:       Draw as directed or as undirected graph.
        seed:           Random seed used to determine the node positions.
        check_overlap:  If true, ensure that the overlap projection magnitude does not exceed 1.
        ax:             Draw the overlap graph in the specified Matplotlib axes.

    Raises:
        ValueError:     invalid input
    """
    nmo1, nmo2 = S.shape
    if mo_lists is None:
        mo_lists = (np.arange(nmo1), np.arange(nmo2))
    if labels is None:
        labels = (np.arange(nmo1), np.arange(nmo2))

    # Square of each element in the overlap matrix.
    S2 = np.square(S)

    # For orthonormal orbitals, the sum of each row or each column must not exceed 1.
    for axis in 0, 1:
        if check_overlap and np.any(np.sum(S2, axis=axis) - 1.0 > 1e-8):
            raise ValueError("Overlap normalized to a value > 1.0.")

    # Set up a graph. We can treat directed and undirected graphs equivalently.
    G = networkx.DiGraph() if directed else networkx.Graph()

    # Iterate over the active rows of the overlap matrix.
    for row_idx in mo_lists[0]:
        row = S2[row_idx, :]
        # Exclude active orbital pairs that map almost one-to-one.
        if np.amax(row[mo_lists[1]]) <= upper_thresh:
            # Exclude small overlaps.
            for col_idx in np.argwhere(row >= lower_thresh).flatten():
                # Edge with weight "overlap squared" between the two orbitals.
                G.add_edge((0, row_idx), (1, col_idx), weight=row[col_idx])

    # Iterate over the active columns of the overlap matrix.
    for col_idx in mo_lists[1]:
        col = S2[:, col_idx]
        # Exclude active orbital pairs that map almost one-to-one.
        if np.amax(col[mo_lists[0]]) <= upper_thresh:
            # Exclude small overlaps.
            for row_idx in np.argwhere(col >= lower_thresh).flatten():
                # Edge with weight "overlap squared" between the two orbitals.
                G.add_edge((0, row_idx), (1, col_idx), weight=col[row_idx])

    # Weights in a format that is useful with networkx...
    weights_dict = networkx.get_edge_attributes(G, "weight")
    weights = np.array(list(weights_dict.values()))

    # Labels for the nodes (orbital indices by default).
    labels_dict = {}
    for g, i in G.nodes:
        labels_dict[(g, i)] = labels[g][i]

    # Assign the colors for the nodes.
    node_colors = []
    for g, i in G.nodes:
        if i in mo_lists[g]:
            node_colors.append(active_colors[g])
        else:
            node_colors.append(inact_colors[g])

    # Widths to draw the edges.
    scaled_edge_widths = edge_width * weights

    # Positions of the nodes.
    pos = networkx.spring_layout(G, seed=seed)

    # Draw the graph.
    networkx.draw(
        G,
        pos=pos,
        node_color=node_colors,
        with_labels=True,
        labels=labels_dict,
        edge_color=edge_color,
        width=scaled_edge_widths,
        ax=ax,
    )

    # Label the edges with the squared overlap values.
    edge_labels = {key: "{:.2f}".format(weight) for key, weight in weights_dict.items()}
    networkx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)


def draw_overlap_with_unrestricted(
    S: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    mo_list: Optional[Union[list[int], np.ndarray]] = None,
    mo_offset: int = 0,
    lower_thresh: float = 0.10,
    upper_thresh: Optional[float] = None,
    colors: tuple[str, str] = ("crimson", "royalblue"),
    **kwargs,
) -> None:
    """Represent the overlap between a restricted and an unrestricted orbital set graphically.

    Args:
        S:              Overlap matrices between the restricted orbital set and each set of
                        unrestricted orbitals. Rows represent the restricted orbitals, which are
                        identical in both matrices. Columns represent the unrestricted orbitals.
        mo_list:        Optionally, subset of restricted orbitals to show (S contains full set).
        mo_offset:      Integer offset to add to all orbital indices in the network graph.
        lower_thresh:   Edge only drawn if the square of the overlap exceeds this threshold.
        upper_thresh:   Omit edges with the square of the overlap above this threshold.
        colors:         Colors to use for the restricted (1st) and unrestricted nodes (2nd).
        **kwargs:       Further keyword arguments for the drawing functions.

    Raises:
        ValueError:     Bad input.
    """
    # Convert S to array and check for correct dimensionality.
    S = np.array(S)
    if S.ndim != 3 or S.shape[0] != 2:
        raise ValueError("S must consist of two overlap matrices.")

    # Set default mo_list or convert to array.
    if mo_list is None:
        mo_list = np.arange(S.shape[1])
    mo_list = np.array(mo_list)

    # By default, use a dummy value to avoid cutting off large overlaps.
    if upper_thresh is None:
        upper_thresh = 2.0

    # Merge columns of the alpha and beta overlap matrices.
    S_merged = np.concatenate((S[0][mo_list, :], S[1][mo_list, :]), axis=1)

    # Labels for the spin-restricted orbitals.
    labels_restricted = mo_list + mo_offset

    # Labels for the spin-unrestricted orbitals integer index plus a or b.
    labels_unrestricted = []
    for spin_label in ("a", "b"):
        for i in range(S.shape[2]):
            labels_unrestricted.append(f"{i + mo_offset}{spin_label}")
    draw_overlap_active(
        S=S_merged,
        labels=(labels_restricted, labels_unrestricted),
        lower_thresh=lower_thresh,
        upper_thresh=upper_thresh,
        active_colors=colors,
        check_overlap=False,
        **kwargs,
    )


def scale01(x: Union[np.ndarray, list[float]], xhalf: float, exponent: float) -> np.ndarray:
    """Rescales each entry of x to a value between 0 and +1 or -1.

    The function is f(x) = sign(x) * 1 / [1 + (x0 / |x|) ^ k].
    Signs of the values in x are preserved.

    Args:
        x:          The array containing the values to be scaled.
        xhalf:      The value x0 > 0, such that f(x0) = 0.5.
        exponent:   The exponent k > 0.

    Returns:
        An array with the scaled values.

    Raises:
        ValueError: Invalid input
    """
    if xhalf <= 0.0:
        raise ValueError("xhalf must be positive")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    xarr = np.array(x)
    scaledx = np.zeros_like(xarr)
    ratio = np.abs(xhalf / xarr[xarr != 0])
    scaledx[xarr != 0] = 1 / (1 + ratio**exponent)
    scaledx[xarr < 0] = -scaledx[xarr < 0]
    return scaledx


def draw_pair_information(
    pair_info: np.ndarray,
    pair_half_val: float = 0.03,
    pair_exponent: float = 1.5,
    orbital_info: Optional[np.ndarray] = None,
    orbital_half_val: float = 0.05,
    orbital_exponent: float = 1.5,
    orbital_labels: Optional[Union[np.ndarray, list]] = None,
    edge_width: float = 5.0,
    edge_cmap: str = "seismic",
    node_cmap: str = "RdPu",
    seed: int = 0,
    verbose: bool = True,
    ax: Optional[Axes] = None,
) -> None:
    """Represents a network of pair information graphically.

    Arguments:
        pair_info:          The pair information, typically minus the cumulant.
        pair_half_val:      Magnitude at which an edge representing pair information is shown
                            with half its maximum width and intensity.
        pair_exponent:      The larger the exponent, the quicker the change from no to full color
                            saturation and width of each edge.
        orbital_info:       Information to color the nodes (representing orbitals). If None is
                            provided, use the pair information sum for each orbital.
        orbital_half_val:   Value for which to color the nodes with half their maximal saturation.
        orbital_exponent:   The larger the exponent, the quicker the change from no to full
                            color saturation of each node.
        orbital_labels:     Labels to use for the nodes / orbitals. By default, use their indices
                            starting with zero.
        edge_width:         Maximum width to draw the edges.
        edge_cmap:          Color map for the edges. Assuming that the pair information can be
                            positive and negative, a "diverging" color map should be used.
        node_cmap:          Color map for the nodes. Assuming that the values are non-negative,
                            a sequential color map should be used.
        seed:               Random seed to determine the positions of the nodes.
        verbose:            Information printing.
        ax:                 Draw the pair information graph in the specified Matplotlib axes.

    Raises:
        ValueError: Invalid input
    """
    nmo = pair_info.shape[0]
    if pair_info.shape != (nmo, nmo):
        raise ValueError("pair_info array must have shape (nmo, nmo).")

    if orbital_info is None:
        orbital_info = np.sum(pair_info, axis=0) + np.sum(pair_info, axis=1) - np.diag(pair_info)
    elif orbital_info.shape != (nmo,):
        raise ValueError("orbital_info must be a vector of length nmo.")

    if orbital_labels is None:
        orbital_labels = np.arange(nmo)
    elif len(orbital_labels) != nmo:
        raise ValueError("orbital_labels must be of length nmo.")

    # create graph with the pair data
    G = networkx.Graph()
    for p in range(nmo):
        G.add_edge(p, p, weight=pair_info[p, p])
        for q in range(p):
            xpq = pair_info[p, q] + pair_info[q, p]
            G.add_edge(p, q, weight=xpq)

    weights_dict = networkx.get_edge_attributes(G, "weight")

    # graph with unsigned pair data to arrange the node positions
    Gabs = networkx.Graph()
    for (p, q), weight in weights_dict.items():
        Gabs.add_edge(p, q, weight=abs(weight))

    # determine node positions using the absolute values of the weights
    pos = networkx.spring_layout(Gabs, seed=seed)

    if verbose:
        print("      MO    orb. value")
        for i in range(nmo):
            print("{0:8}      {1:8.4f}".format(orbital_labels[i], orbital_info[i]))

    # draw the graph
    label_dict = {i: orbital_labels[i] for i in range(nmo)}
    scaled_node_colors = scale01(orbital_info, orbital_half_val, orbital_exponent)
    scaled_weights = scale01(list(weights_dict.values()), pair_half_val, pair_exponent)
    scaled_edge_widths = edge_width * np.abs(scaled_weights)
    scaled_edge_colors = 0.5 * scaled_weights + 0.5
    networkx.draw(
        G,
        pos=pos,
        with_labels=True,
        labels=label_dict,
        node_color=scaled_node_colors,
        cmap=colormaps[node_cmap],
        width=scaled_edge_widths,
        edge_color=scaled_edge_colors,
        edge_cmap=colormaps[edge_cmap],
        ax=ax,
    )


def draw_diagonal_cumulant(
    rdm1a: np.ndarray, rdm1b: np.ndarray, rdm2: np.ndarray, show_entropy: bool = True, **kwargs
) -> None:
    """Draw a network representing cumulant information calculated from reduced density matrices.

    Args:
        rdm1a:          Spin-up part of the one-particle reduced density matrix.
        rdm1b:          Spin-down part of the one-particle reduced density matrix.
        rdm2:           Spin-free two-particle reduced density matrix.
        show_entropy:   Use entropy to color the nodes if set to true. Otherwise, color nodes based
                        on the cumulant information.
        **kwargs:       Keyword arguments for draw_pair_information(...).
    """
    # The full cumulant.
    cumulant_full = cumulant2b(rdm1a, rdm1b, rdm2)

    # Extract the diagonal elements of the cumulant (no sum).
    cdiag = np.einsum("ppqq->pq", cumulant_full)

    # Calculate the entropy if this is requested to color the nodes.
    args_dict = kwargs.copy()
    if show_entropy:
        orbdens = orbdens_from_rdm(rdm1a, rdm1b, 0.5 * rdm2)
        entropy = calculate_entropy_s1(orbdens)
        args_dict["orbital_info"] = entropy
        if "orbital_half_val" not in args_dict:
            # Adjust the coloring threshold for the entropy.
            args_dict["orbital_half_val"] = 0.139

    # Correlation partner orbitals will have negative cumulant entries, so invert the sign.
    draw_pair_information(-cdiag, **args_dict)


def draw_pair_information_uhf(
    pair_info: Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]],
    orbital_info: Optional[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]] = None,
    orbital_indices: Optional[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]] = None,
    **kwargs,
) -> None:
    """Draw pair information for spin-unrestricted orbitals in one graph.

    Args:
        pair_info:          Matrices with pair information for spin orbitals: (aa, ab, bb)
        orbital_info:       Single-orbital information for alpha and beta MOs: (alpha, beta)
        orbital_indices:    Integer indices of alpha and beta MOs: (alpha indices, beta indices)
        **kwargs:           Further keyword arguments to pass to draw_pair_information.

    Raises:
        ValueError: Invalid input.
    """
    # Sanity checking of spin orbital pair info.
    if len(pair_info) != 3:
        raise ValueError(f"pair_info with wrong leading dimension: {len(pair_info)}")

    # Numbers of alpha and beta spin orbitals.
    Na = pair_info[0].shape[0]
    Nb = pair_info[2].shape[0]

    # More sanity checking of spin orbital pair info.
    for s, N1, N2 in ((0, Na, Na), (1, Na, Nb), (2, Nb, Nb)):
        if pair_info[s].shape != (N1, N2):
            raise ValueError(f"pair_info[{s}] with shape {pair_info[s].shape} != ({N1}, {N2})")

    # Sanity checking of single-orbital information.
    if orbital_info is not None:
        if len(orbital_info) != 2:
            raise ValueError(f"orbital_info with wrong leading dimension: {len(orbital_info)}")
        for s in (0, 1):
            if orbital_info[s].ndim != 1:
                raise ValueError(f"orbital_info[{s}] with wrong shape: {orbital_info[s].shape}")

    # Fill orbital indices with integers from 0 to Na or Nb if unspecified.
    if orbital_indices is None:
        orbital_indices = (np.arange(Na), np.arange(Nb))

    # Sanity checking of orbital index lists.
    if len(orbital_indices) != 2:
        raise ValueError(f"orbital_indices with wrong leading dimension: {len(orbital_indices)}")
    for s in (0, 1):
        if orbital_indices[s].ndim != 1:
            raise ValueError(f"orbital_indices[{s}] with wrong shape: {orbital_indices[s].shape}")

    # Merging pair information blocks into a single matrix spanning both alpha and beta orbitals.
    merged_pairinfo = np.block([[pair_info[0], pair_info[1]], [pair_info[1].T, pair_info[2]]])

    # Merging single-orbital information into a single vector.
    if orbital_info is not None:
        merged_orbinfo = np.concatenate((orbital_info[0], orbital_info[1]))
    else:
        merged_orbinfo = None

    # Creating merged list of orbital labels "[index]a" or "[index]b" for the respective spins.
    merged_labels = [f"{i}a" for i in orbital_indices[0]] + [f"{i}b" for i in orbital_indices[1]]

    # Pass everything to the function to draw pair information.
    draw_pair_information(
        pair_info=merged_pairinfo,
        orbital_info=merged_orbinfo,
        orbital_labels=merged_labels,
        **kwargs,
    )

    return
