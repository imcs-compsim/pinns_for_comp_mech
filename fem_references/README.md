# FEM references

This folder contains Git submodules used by the main repository for specific reference results obtain through FEM simulations to compare the PINN results to.

## What is a submodule?

A submodule is a reference from the main repository to a specific commit of another repository.

That means:
- the code contained in the subdirectories of this folder belongs to separate Git repositories
- the main repository only stores pointers to a specific commit of those repositories
- updating a submodule requires committing changes in both the submodule and the main repository

## Cloning the repository

To clone the main repository together with all submodules:

```bash
$ git clone --recurse-submodules <https://github.com/imcs-compsim/pinns_for_comp_mech.git>
```
If you already cloned the repository without the submodules, you can get them by running the following command in the top-level folder:
```bash
$ git submodule update --init --recursive
```
## Commit, pushing and pulling
As the main repo and the submodules are separate repositories you also have to take care of their respective commit structures and branches. This explicitly means that when changing something in the submodule you have to update the reference of the main repository to point to the respective commit hash of the submodule repository. To do so
```bash
$ git pull --recurse-submodules
```
