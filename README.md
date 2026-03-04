# Physics-Informed Neural Networks for Computational Mechanics (compsim_pinns)
```
 ██████╗ ██████╗ ███╗   ███╗██████╗ ███████╗██╗███╗   ███╗        ██████╗ ██╗███╗   ██╗███╗   ██╗███████╗
██╔════╝██╔═══██╗████╗ ████║██╔══██╗██╔════╝██║████╗ ████║        ██╔══██╗██║████╗  ██║████╗  ██║██╔════╝
██║     ██║   ██║██╔████╔██║██████╔╝███████╗██║██╔████╔██║        ██████╔╝██║██╔██╗ ██║██╔██╗ ██║███████╗
██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ╚════██║██║██║╚██╔╝██║        ██╔═══╝ ██║██║╚██╗██║██║╚██╗██║╚════██║
╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ███████║██║██║ ╚═╝ ██║███████╗██║     ██║██║ ╚████║██║ ╚████║███████║
 ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝
```
--------------------------------------------------------------
  A Multipurpose Python Framework for Computational Mechanics
  based on Physics-Informed Neural Networks (PINNs)
--------------------------------------------------------------

**CompSim_PINNs** is a Python framework originated and maintained by the [Institute for Mathematics and Computer-Based Simulation (IMCS)](https://www.unibw.de/imcs-en) at the [University of the Bundeswehr Munich](https://www.unibw.de/home-en).

This framework implements the functionalities of PINNs (Physics-Informed Neural Networks) using the [`DeepXDE`](https://deepxde.readthedocs.io/en/latest/) package for solid and contact mechanics.

> **Note**: This framework and and its documentation are under active development.

### Examples provided by the framework

- Euler-Bernoulli beams
  - Dynamic beam equation
  - Static beam equation
- Heat equation problems
- Linear elasticity
  - Four-point bending test
  - Forward and inverse Lamé problem
  - Solid beam
  - 3D hollow sphere subjected to internal pressure
  - 4D problem: 3D hollow sphere under time-dependent loading
- Contact problems
  - 2D contact between an elastic block and a rigid surface
  - 2D Hertzian contact problem
  - 3D single contact patch test
  - 3D cylindrical contact problem
- Large deformation (Deep-energy methods)
  - Solid mechanics
    - Bending beam under shear load (2D)
    - Lamé problem
    - 3D torsion of a square prism
  - Contact mechanics (single-step and incremental loading approaches)
    - Single patch test (2D and 3D)
    - 2D Hertzian contact problem
    - 2D Contact ring example (2D)
    - 3D spherical contact problem
    - 3D torus contact instability problem
---
## Installation of the DeepXDE package and required libraries

This framework relies on the `deepxde` package for training PINNs.

> **Note**: If you want to be able to debug your PINN training code and step
> into functions provided by `deepxde`, you might want to skip the following
> instructions and instead install it from source in editable mode.
> Read [their website](https://deepxde.readthedocs.io/en/latest/user/installation.html)
> for instructions on how to do that.

`deepxde` needs one of the following packages for the backend-calculation.
- PyTorch (preferred backend by this framework)
- TensorFlow (not supported by all examples, remaining support within this framework might be dropped in the future)

### Setup with `conda`
This repository also comes with an `env.yaml` file to directly create a `conda` environment with all dependencies.
The provided `conda` environment is configured to include the development dependencies and use `tensorflow` as backend for PINN training.
Here we leverage an installation via `conda-forge` to be able to install specific versions that are tailored to the available hardware.
To create an environment, run
```bash
$ conda env create -f env.yaml
```
in the top-level repository folder after cloning.

### Setup with `pip`
You can specify your backend of choice when setting up this framework by running
* for Pytorch
```bash
$ pip install -e ".[torch]"
```
* for Tensorflow
```bash
$ pip install -e ".[tf]"
```
in the top-level repository folder after cloning.

If you additionally want to install packages for development (i.e., for running unittests or building the documentation), you can do so by additionally selecting the `dev` configuration, e.g.,
```bash
$ pip install -e ".[torch,dev]"
```

---

## Testing

This repo has `integration_tests` (testing for examples/frameworks) and `unittests` (testing for specific functions).
Testing is done by `pytest` and tests are configured in the `pyproject.toml` file.

To run tests, type in the terminal:
```bash
$ pytest
```

---

## Pre-commit hooks

This repo uses `pre-commit` to enforce formatting, linting, type checks, and commit message conventions.

Install the git hooks:
```bash
$ pre-commit install
$ pre-commit install --hook-type commit-msg
```

---

## Docker image

This framework provides its functionality also in a Docker. To use it that way follow these instructions:\
First, build the docker container using `Dockerfile` and type the following command on the terminal
```bash
docker build -t imcs-pinn -f docker/Dockerfile .
```

Then, run the docker container by typing the following command on the terminal
```bash
docker run imcs-pinn
```
`NOTE`: The command above will automatically run the default commands defined in Dockerfile.

When you want to open an interactive shell inside the container or interact with the process manually, for instance for debugging, then run
```bash
docker run -it imcs-pinn bash
```

Run, a specific example
```bash
docker run -it imcs-pinn conda run -n compsim_pinns python examples/elasticity_3d/linear_elasticity/block_under_shear.py
```

# Citing 'compsim_pinns'

Whenever you use or mention 'compsim_pinns' in some sort of scientific document/publication/presentation, please cite the following publications. They are publicly available at [AMSES](https://amses-journal.springeropen.com/articles/10.1186/s40323-024-00265-3) and [Springer Nature](https://link.springer.com/chapter/10.1007/978-3-031-93213-7_33).

```
@article{Sahin2024,
  title = {Solving Forward and Inverse Problems of Contact Mechanics Using Physics-Informed Neural Networks},
  author = {Sahin, Tarik and Von Danwitz, Max and Popp, Alexander},
  year = {2024},
  journal = {Advanced Modeling and Simulation in Engineering Sciences},
  volume = {11},
  number = {1},
  pages = {11},
  issn = {2213-7467},
  doi = {10.1186/s40323-024-00265-3},
  date = {2024-05-08}
}

@incollection{Sahin2025,
  title = {Physics-Informed Neural Networks for Solving Contact Problems in Three Dimensions},
  booktitle = {Advances and Challenges in Computational Mechanics},
  author = {Sahin, Tarik and Wolff, Daniel and Popp, Alexander},
  editor = {Graf, Wolfgang and Fleischhauer, Robert and Storm, Johannes and Wollny, Lines},
  date = {2025},
  pages = {419--431},
  publisher = {Springer Nature Switzerland},
  location = {Cham},
  doi = {10.1007/978-3-031-93213-7_33},
  isbn = {978-3-031-93212-0 978-3-031-93213-7}
}
```
