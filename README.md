# Active Space Finder (ASF)
The Active Space Finder (ASF) is a set of functions for the automatic/semi-automatic
selection of active spaces in molecules. The selection can either be performed with minimal manual intervention
through a wrapper function that was designed to execute a general workflow;
alternatively, you can design your own procedure using the individual components.
The code is developed at `HQS Quantum Simulations <https://quantumsimulations.de/>`_,
and it is released under the `Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

ASF requires the following software packages:

- Python3 with numpy and scipy
- `PySCF v.1.7 <https://github.com/pyscf/pyscf/>`_ for general electronic structure calculations
- `BLOCK v.1.5 <https://github.com/sanshar/Block/>`_ for DMRG calculations
- `Jmol v.14.6.4 <http://jmol.sourceforge.net/>`_ for plotting

ASF is developed with the following principles in mind:

- Automation
- Flexibility
- Ease of development

ASF follows the coding standards:

- flake8
- mypy

The ASF can completely automatize the selection of an active space, thus facilitating the black-box application
of multi-reference methods such as CASSCF. The automated
wrapper function selects an initial large active space using natural orbitals
obtained through the user's method of choice (MP2, CCSD, ...). The large active space is then used for DMRG-CI
calculations, which provide single-orbital entropies that are used to select the final active space.
The procedure for selection can be modified by the user. For instance, one
can select the final orbitals after the DMRG-CI calculation based on natural orbitals
instead of entropies.

Please be aware of the following difficulties in active space selection:

- Selecting orbitals, especially beyond a highly entangled, minimal active space, is not an easy task.
  This is particularly true for single-reference systems (no orbital degeneracy).
- Single-orbital entropies can be influenced strongly by the orbital choice,
  as a unitary transformation with single excitations is equivalent to an orbital rotation.
- DMRG calculations starting from very small M values can converge to an excited state.


Finally, CASSCF without a further treatment of dynamic correlation is rarely suited for quantitative purposes,
and therefore it usually cannot reproduce numbers close to experiment. Perturbation-theory based methods such as NEVPT2 and CASPT2 remedy this problem
to some extent. However, perturbation theory (at second order) does not always recover the dynamic correlation energy
sufficiently accurately for practical applications, for instance when calculating
spin state energetics. A common practice in such cases is to extend the active space beyond a minimal size,
so that a larger portion of the dynamic correlation energy is recovered already by CASSCF. Constructing such active spaces may require an
increased amount of both chemical intuition and trial and error, which is not always amenable to be automated straightforwardly with generic criteria.

In order to demonstrate the performance of the ASF, we provide a small set of :ref:`benchmark calculations`.
These calculations focus on vertical excitation/transition energies of small/medium sized organic molecules.

If you find the active space finder a useful tool for your research, please cite our work.

Documentation + Tutorials
--------------------------

[*click*](https://ActiveSpaceFinder.readthedocs.io/en/latest/)
