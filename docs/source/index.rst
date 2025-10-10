.. CompSim-PINN documentation master file, created by
   sphinx-quickstart on Thu Oct  9 16:00:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the **CompSim-PINN** documentation !
===============================================

**CompSim-PINN** is a Python framework origniated and maintained by the `Institute for Mathematics and Computer-Based Simulation (IMCS) <https://www.unibw.de/imcs-en>`_ at the `University of the Bundeswehr Munich <https://www.unibw.de/home-en>`_.

This framework implements the functionalities of :abbr:`PINNs (Physics-Informed Neural Networks)` using the `DeepXDE <https://deepxde.readthedocs.io/en/latest/>`_ package for solid and contact mechanics.

Usage
=====

.. _installation:

Installation
------------

To use CompSim-PINN, first clone it into your local repository:

.. code-block:: console

   $ git clone https://github.com/imcs-compsim/pinns_for_comp_mech.git

After this you can directly set up a ``conda`` environment that uses ``torch`` as a backend in ``deepxde``:

.. code-block:: console

   $ conda env create -f env.yaml

.. note::
   This framework is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules