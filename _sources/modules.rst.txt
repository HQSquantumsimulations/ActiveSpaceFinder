Code documentation
===================


.. autoclass:: asf.ASFDMRG
   :members:
   :inherited-members:
.. autoclass:: asf.ASFCI
   :members:
   :inherited-members:

``asf.asfbase``
---------------
The classes and functions contained in the ``asfbase`` module fulfill a more fundamental role. For
instance, the ``ActiveSpace`` class is a representation of an active space which is often used as
a return type. Another notable function is ``print_mo_table`` which shows one-orbital densities and
the one-orbital entropy for selected molecular orbitals.

.. automodule:: asf.asfbase
   :members:


``asf.filters``
---------------
The ``filters`` module implements functions to filter or select from collections of active spaces.
Apart from ``find_sensible`` which determines a set of reasonable active spaces, ``FilterFunction``
and ``SelectorFunction`` are useful objects, e.g. for defining custom filters or selectors.

.. automodule:: asf.filters
   :members:


``asf.molmath``
---------------

.. automodule:: asf.molmath
   :members:


``asf.mp2density_conventional``
-------------------------------

.. automodule:: asf.mp2density_conventional
   :members:


``asf.natorbs``
---------------

.. automodule:: asf.natorbs
   :members:


``asf.pairinfo``
----------------
The ``pairinfo`` module implements the ``PairInfoAnalyzer`` which computes the metrics which form
the basis of selection process. These metrics are collected in the ``MOListInfo`` dataclass and
the ``ActiveSpacesDict`` type, both contained in this module.

.. automodule:: asf.pairinfo
   :members:
   :exclude-members: cumulant2b, cumulant2bs, transform_tensor, transform_unrestricted_rdm12s, transform_unrestricted_rdm1s, transform_unrestricted_rdm2s, somo_from_orbdens 


``asf.preselection``
--------------------

.. automodule:: asf.preselection
   :members:


``asf.rimp2_pairinfo``
----------------------

.. automodule:: asf.rimp2_pairinfo
    :members:


``asf.scf``
-----------

.. automodule:: asf.scf
    :members:


``asf.utility``
---------------
Besides the general functions contained in the ``utility`` module, ``pictures_Jmol`` and
``write_jmol_script`` are especially relevant for visualizing an active space.

.. automodule:: asf.utility
    :members:
    :exclude-members: loghead, loginfo


``asf.visualization``
---------------------

.. automodule:: asf.visualization
    :members:
    :exclude-members: scale01


``asf.wrapper``
---------------

.. automodule:: asf.wrapper
    :members:
