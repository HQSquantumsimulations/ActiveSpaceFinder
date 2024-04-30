Installation
============

Requirements
------------

We recommend Python versions ``3.9`` to ``3.11``. Please note that ``block2=0.5.2`` has not been packaged
for Python newer than ``3.11`` on `pypi.org <https://pypi.org/project/block2/>`_.
The Active Space Finder's main dependencies [#]_ are the following software packages:

- `numpy <https://numpy.org/>`_ and `scipy <https://scipy.org/>`_
- `PySCF <https://github.com/pyscf/pyscf/>`_ for general electronic structure calculations
- `block2 <https://github.com/block-hczhai/block2-preview/>`_ for DMRG calculations
- `matplotlib <https://matplotlib.org/>`_ and `networkx <https://networkx.org/>`_ for visualization of ASF metrics
- `Jmol <https://jmol.sourceforge.net/>`_ for visualization of molecular orbitals


Installation from source
------------------------

After checking out the repository, a local installation of the ASF package from the source directory requires two steps:

.. sourcecode:: sh

    pip install .
    ./init_dmrgscf_settings.sh

First, the dependencies are installed via ``pip``. Because the ASF code is built upon `PySCF <https://github.com/pyscf/pyscf>`_ and `Block2 <https://github.com/block-hczhai/block2-preview>`_ for DMRG calculations, a configuration file for PySCF's ``dmrgscf`` plugin is created in a second step. For convenience, the ASF package includes a script to create the configuration file automatically (run ``./init_dmrgscf_settings.sh --help`` to learn more about its usage).

The examples directory contains Jupyter notebooks. To work with those, you may need to install
Jupyter Notebook or a similar software.

----

.. [#] For a full list of dependencies please check ``pyproject.toml`` in the root directory of the `Active Space Finder repository <https://github.com/HQSquantumsimulations/ActiveSpaceFinder>`_.
