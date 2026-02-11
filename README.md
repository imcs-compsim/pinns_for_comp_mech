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
### Included examples
---
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
- PyTorch (preferred backend)
- TensorFlow (not supported by all examples, remaining support might be dropped in the future)

You can specify your backend of choice when seting up this framework by running
* for Tensorflow
```bash
$ pip install -e ".[tf]"
```
* for Pytorch
```bash
$ pip install -e ".[torch]"
```
in the top-level repository folder after cloning.

If you additionally want to install packages for development (i.e., for running unittests or buidling the documentation), you can do so by additionally selecting the `dev` configuration, e.g., 
```bash
$ pip install -e ".[tf,dev]"
```

### Setup with `conda` 
This repository also comes with an `env.yaml` file to directly create a `conda` environment with all dependencies. 
The provided `conda` environment is configured to include the development dependencies and use `tensorflow` as backend for PINN training.
Here we leverage an installation via `conda-forge` to be able to install specific versions that are tailored to the available hardware.
To create an environment, run 
```bash
$ conda env create -f env.yaml
```
in the top-level repository folder after cloning.

---

## Testing

This repo has `integration_tests` (testing for examples/frameworks) and `unittests` (testing for specific functions). 
Testing is done by `pytest` and tests are configured in the `pyproject.toml` file. 

To run tests, type on the terminal:
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

Run hooks on the entire repository (first-time setup)
```bash
$ pre-commit run --all-files
```

---

## Cluster setup
For cluster, we should use `conda` since we had issues in terms of package installation particularly the package `gmsh`.  Enable pinn repo to run on cluster:

1. Install miniconda https://docs.conda.io/en/latest/miniconda.html :

    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
2. Create the virtual environment using `env.yaml` which includes all necessary packages. 
    ```bash
    conda env create -f env.yaml
    ```

3. Activate the generated venv (`compsim_pinns`) 
    ```bash
    conda activate compsim_pinns
    ```
4. To test cluster, submit a job on a compute node. This is achieved through `test_cluster.sh` (full path: pinnswithdxde/tests/integration_tests/cluster/test_cluster.sh).

    ```bash
    $ sbatch $HOME/pinnswithdxde/tests/integration_tests/cluster/test_cluster.sh
    ```
    Number of threads is set in `test_cluster.sh` file. TensorFlow needs to be `intra_op_parallelism_threads` and `inter_op_parallelism_threads` parameters set. Thus, we give  `tf_cluster_settings.py` to the slurm job via sbatch. This enables TensorFlow to set OMP parameters that defined in `test_cluster.sh`.

> `NOTE`: Do not forget to adopt the inside of the `test_cluster.sh` to specify the slurm options e.g., `--mail-user`. But the default one should work without error. 

> `NOTE`: Always be sure that you activated venv `compsim_pinns` (step 3) before `sbatch` any slurm script. This includes other scripts you will run as well. The reason behind is that activating venv in `test_cluster.sh` needs the full path for the conda env `compsim_pinns` and it gives some **init** error if the full path is used.  

> `NOTE`: For conda commands: A conda [cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) can be very useful. 

> `NOTE`: Some usefull information regarding [CPU](https://github.com/PrincetonUniversity/slurm_mnist/tree/master/cpu_only#readme) on cluster. 

---

## How to use docker?

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

## Citing 'compsim_pinns'

Whenever you use or mention 'compsim_pinns' in some sort of scientific document/publication/presentation, please cite the following publications. They are publicly avaliable at [AMSES](https://amses-journal.springeropen.com/articles/10.1186/s40323-024-00265-3) and [ArXiv](https://arxiv.org/abs/2412.09022).

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
  editor = {Graf, Wolfgang and Fleischhauer, Robert and Storm, Johannes and Wollny, Ines},
  date = {2025},
  pages = {419--431},
  publisher = {Springer Nature Switzerland},
  location = {Cham},
  doi = {10.1007/978-3-031-93213-7_33},
  isbn = {978-3-031-93212-0 978-3-031-93213-7}
}
```

Paper results are obtained in hastag: c79d3f24023e36341385f10d728e5a93c925fad3
