# Physics-Informed Neural Networks for Computational Mechanics (pinns_for_comp_mech)

Implementation of Physics-Informed Neural Networks (PINNs) for computational mechanics based on the DeepXDE package.

- Euler-Bernoulli beams
- Heat equation problems
- Linear elasticity
  - Four-point bending test
  - Forward and inverse Lame` problems
  - Solid beam
- Contact problems
  - Contact between elastic body and rigid flat surface
  - Hertzian contact problem
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

3. Activate the generated venv (`pinns-for-comp-mech`) 
    ```bash
    conda activate pinns-for-comp-mech
    ```
4. To test cluster, submit a job on a compute node. This is achieved through `test_cluster.sh` (full path: pinnswithdxde/tests/integration_tests/cluster/test_cluster.sh).

    ```bash
    $ sbatch $HOME/pinnswithdxde/tests/integration_tests/cluster/test_cluster.sh
    ```
    Number of threads is set in `test_cluster.sh` file. TensorFlow needs to be `intra_op_parallelism_threads` and `inter_op_parallelism_threads` parameters set. Thus, we give  `tf_cluster_settings.py` to the slurm job via sbatch. This enables TensorFlow to set OMP parameters that defined in `test_cluster.sh`.

> `NOTE`: Do not forget to adopt the inside of the `test_cluster.sh` to specify the slurm options e.g., `--mail-user`. But the default one should work without error. 

> `NOTE`: Always be sure that you activated venv `pinns-for-comp-mech` (step 3) before `sbatch` any slurm script. This includes other scripts you will run as well. The reason behind is that activating venv in `test_cluster.sh` needs the full path for the conda env `pinns-for-comp-mech` and it gives some **init** error if the full path is used.  

> `NOTE`: For conda commands: A conda [cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) can be very useful. 

> `NOTE`: Some usefull information regarding [CPU](https://github.com/PrincetonUniversity/slurm_mnist/tree/master/cpu_only#readme) on cluster. 

---

## Citing 'pinns_for_comp_mech'

Whenever you use or mention 'pinns_for_comp_mech' in some sort of scientific document/publication/presentation, please cite the following publication. It is publicly avaliable at [arXiv](https://arxiv.org/abs/2308.12716).

```
@article{sahin2023pinnforcontact,
   title={Solving Forward and Inverse Problems of Contact Mechanics using Physics-Informed Neural Networks},
   author={Sahin, Tarik and von Danwitz, Max and Popp, Alexander},
   journal={arXiv preprint arXiv:2308.12716},
   year={2023}
}
```

Paper results are obtained in hastag: c79d3f24023e36341385f10d728e5a93c925fad3
