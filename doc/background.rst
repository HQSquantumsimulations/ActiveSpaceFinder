Background
==========

Overview
--------

This section provides an overview of the principles underlying active space selection with this
software package, but it does not attempt a detailed explanation; this will be reserved for a
forthcoming publication.

The most important part of a selection performed with the ASF involves a DMRG calculation. It is
performed with low-accuracy settings that afford a low computational cost. These results are not
suited for quantitative studies, but they contain sufficient qualitative information for an
active space selection. In this regard, our approach is somewhat similar to the procedure described
by Stein and Reiher. [1]_ Otherwise, the procedure employs different principles.

Orbital pre-selection
---------------------

Since the cost of a DMRG calculation increases very steeply with the number of orbitals, an initial
active space needs to be determined in a first step. We normally employ Møller-Plesset perturbation
theory for this purpose. The ASF integrates two variants:

- Pre-selection with MP2 natural orbitals and their associated occupation numbers. This employs
  largely PySCF functionality via the
  `DF-MP2 method <https://pyscf.org/user/mp.html#density-fitted-mp2-df-mp2>`_.
- Pre-selection using Møller-Plesset pair information implemented in the ASF. Compared with MP2
  natural orbitals, this is a more selective method as it is better at identifying correlation
  partners, but it is also more expensive.

Hence, the trade-off between the two pre-selection alternatives is either a less expensive
calculation, but a potentially larger active space for DMRG (MP2 natural orbitals); or a more
expensive screening step (perturbative pair information), but potentially a more compact space for
the DMRG calculation.

In either case, Hartree-Fock orbitals need to be determined first. For an open-shell system, the
only option is unrestricted Hartree-Fock (UHF). Restricted Hartree-Fock (RHF) or UHF can be used
for closed-shell systems, but we recommend to *always* use UHF: the symmetry breaking in strongly
or even moderately correlated system helps with active space selection. We provide a convenience
function in the ASF that employs the stability analysis implemented in PySCF to drive UHF
calculations towards a broken-symmetry solution.

Active space selection with DMRG
--------------------------------

The central part of the selection procedure in the ASF is "pair information" representing the
correlation between orbital pairs. It is represented by the two-electron cumulant - the difference
between the two-electron reduced density matrix
(:math:`d_{pqrs} = \langle\Psi|\hat{a}^{\dagger}_{p}\hat{a}^{\dagger}_{r}\hat{a}^{}_{s}\hat{a}^{}_{q}|\Psi\rangle`)
and the antisymmetrized product of the one-electron reduced density matrix elements
(:math:`D_{pq} = \langle\Psi|\hat{a}^{\dagger}_{p}\hat{a}^{}_{q}|\Psi\rangle`):

.. math::

   \Lambda_{pqrs} = \langle \hat{a}^{\dagger}_{p}\hat{a}^{\dagger}_{r}\hat{a}^{}_{s}\hat{a}^{}_{q}\rangle
                  - \langle \hat{a}^{\dagger}_{p}\hat{a}^{}_{q}\rangle\langle\hat{a}^{\dagger}_{r}\hat{a}^{}_{s}\rangle
                  + \langle \hat{a}^{\dagger}_{p}\hat{a}^{}_{s}\rangle\langle\hat{a}^{\dagger}_{r}\hat{a}^{}_{q}\rangle

This quantity is well-established in the literature to quantify the strength of electron
correlation. [2]_ [3]_ [4]_ In particular, we use the semi-diagonal elements, summed over the
spin components of spatial orbitals:

.. math::

   \Lambda_{ppqq} + \Lambda_{pp\bar{q}\bar{q}} + \Lambda_{\bar{p}\bar{p}qq}
   + \Lambda_{\bar{p}\bar{p}\bar{q}\bar{q}}

This results in a simple, yet effective quantification of the correlation strength between orbital
pairs. In the ASF it is usually derived from a DMRG calculation with low-accuracy settings, though
it is also possible to run a conventional CASCI calculation; indeed, for small initial spaces CASCI
is the default choice.

The ASF implements a procedure to analyze this resulting pair information, using it to identify
subsets of correlated orbitals. Indeed, strengths of the current implementation of the ASF include
its ability to determine correlation partner orbitals; and its ability to determine multiple active
spaces of varying sizes for a single geometry.

An important auxiliary quantity in the ASF is the single-orbital entropy. It is computed for each
spatial orbital separately. Entropies quantify the degree of entanglement for each orbital *p* as:

.. math::

   S_{p} = - \sum_{\tau}^{-,\uparrow,\downarrow,\uparrow\downarrow} w_{p, \tau} \ln w_{p, \tau}

Here, :math:`w_{p, \tau}` corresponds to the expectation value that a spatial orbital has a
specific type of occupation: vacant (:math:`-`), with a spin up electron only (:math:`\uparrow`),
with a spin down electron only (:math:`\downarrow`), and doubly occupied (:math:`\uparrow\downarrow`).
These are calculated as the diagonal elements of the one-orbital density matrix [5]_ [6]_
(a different quantity than the one-electron density matrix), with values:

.. math::

   w_{p, -} &= 1 - D_{pp} - D_{\bar{p}\bar{p}} + d_{pp\bar{p}\bar{p}}\\
   w_{p, \uparrow} &= D_{pp} - d_{pp\bar{p}\bar{p}}\\
   w_{p, \downarrow} &= D_{\bar{p}\bar{p}} - d_{pp\bar{p}\bar{p}}\\
   w_{p, \uparrow\downarrow} &= d_{pp\bar{p}\bar{p}}

We employ the entropy as an additional important measure for the overall correlation of a specific
orbital.

Using the Active Space Finder
-----------------------------

To get started using the active space finder, we recommend to follow, execute and modify the
Jupyter notebooks provided with the code. The most straightforward way to use the ASF is to employ
one of the wrapper functions provided. They implement the following functionalities:

- ``find_from_mol`` and ``find_from_scf`` perform a complete active space selection starting either
  from a molecule definition, or from a converged Hartree-Fock solution. Inside the function, an
  orbital pre-selection, the CI or DMRG calculation and the analysis thereof are performed. These
  functions provide some rudimentary functionality to determine active spaces for multiple
  electronic states.
- If you know the size of the desired active space, you can also use the functions
  ``sized_space_from_mol`` or ``sized_space_from_scf``. Internally, they determine multiple suitable
  active spaces. If one of them matches the desired size, it is returned; otherwise, an exception
  is raised.

Note that it is *always* necessary to use the MO coefficients returned by the ASF in conjunction
with the selected space. Using an MO list returned by the ASF and applying it to orbitals
determined directly in an SCF calculation, or in any other way than directly through the
appropriate functionality of the ASF, is meaningless.

More advanced features of the ASF can also be accessed with just a few lines of code by using
the underlying classes of the ASF directly. Generally, this will include the following steps:

1. An SCF calculation, most conveniently performed with ``stable_scf``.
2. Initiating orbital screening through the ``MP2NatorbPreselection`` or ``MP2PairinfoPreselection``
   classes.
3. Carrying out the calculation using the ``ASFDMRG`` or ``ASFCI`` classes.

A particularly powerful way to work with the ASF is to use the ``find_many`` method, which returns
a collection of potentially multiple sensible active spaces determined for a molecule. Out of these
suggestions, you can select an appropriate one yourself - something that is highly recommended
when performing active space calculations.

Further remarks
---------------

While the ASF is often a useful tool that greatly simplifies multi-reference calculations, please
be aware of the following difficulties in active space selection:

- Selecting orbitals, especially beyond a highly entangled, minimal active space, is not a
  trivial task.
- Single-orbital entropies are influenced to some degree by the selection of the initial orbital
  space. In particular, the important, entangled partner orbitals need to be included
  in the DMRG calculation.
- Too small bond dimensions in the DMRG calculation can lead to wrong results.

CASSCF or CASCI treat the correlation in a subset of the most important orbitals exactly. However,
they neglect all the remaining electron correlation in the system, which includes the inactive
orbitals. Therefore, CASSCF results tend to be rather of a qualitative than quantitative nature.
To obtain accurate numbers that can be compared to experiment, it is usually necessary to
use a multi-reference method that includes dynamic correlation, for example NEVPT2 or CASPT2.
The choice of an active space may, to some extent, depend on the specifics of the dynamic
correlation method. For example, spin state energetics calculated with perturbative methods,
such as NEVPT2 or CASPT2, may be improved by using larger active spaces than would be necessary
for a minimal description, thereby also including some dynamic correlation energy in the active
space.

.. rubric:: References

.. [1] C. Stein, M. Reiher, *J. Chem. Theory Comput. 12*, 1760-1771 (**2016**), `doi:10.1021/acs.jctc.6b00156 <https://doi.org/10.1021/acs.jctc.6b00156>`_
.. [2] L. Kong, E. Valeev, *J. Chem. Phys. 134*, 214109 (**2011**), `doi:10.1063/1.3596948 <https://doi.org/10.1063/1.3596948>`_
.. [3] M. Hanauer, A. Köhn, *Chem. Phys. 401*, 50-61 (**2012**), `doi:10.1016/j.chemphys.2011.09.024 <https://doi.org/10.1016/j.chemphys.2011.09.024>`_
.. [4] O. Werba, A. Raeber, K. Head-Marsden, D. A. Mazziotti, *Phys. Chem. Chem. Phys. 21*, 23900-23905 (**2019**), `doi:10.1039/C9CP03361K <https://doi.org/10.1039/C9CP03361K>`_
.. [5] K. Boguslawski, P. Tecmer, Ö. Legeza, M. Reiher, *J. Phys. Chem. Lett. 21*, 3129-3135 (**2012**), `doi:10.1021/jz301319v <https://doi.org/10.1021/jz301319v>`_, `arxiv:1208.6586 <https://arxiv.org/abs/1208.6586>`_
.. [6] K. Boguslawski, P. Tecmer, *Int. J. Quantum Chem. 115*, 1289-1295 (**2015**), `doi:10.1002/qua.24832 <https://doi.org/10.1002/qua.24832>`_
