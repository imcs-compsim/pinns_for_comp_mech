# PINNsWithDXDE

Collection of scripts for our PINN examples with deepXDE

## Installation of the DeepXDE package and required libraries

 ```bash
   $ pip install -r requirements.txt
   ```

For more info: [deepxde website](https://deepxde.readthedocs.io/en/latest/user/installation.html)

Note: DeepXDE needs one of the following packages for the backend-calculation. Read the website for more info. 

- Tensorflow
- Pytorch

## Testing

This repo has `integration_tests` (testing for examples/frameworks) and `unittests` (testing for specific functions). Testing is done by `pytest` and tests are configured in the `setup.cfg` file. 

To run tests, type on the terminal:
```bash
$ pytest
```

## Cluster setup
For cluster, we should use `conda` since we had issues in terms of package installation particularly the package `gmsh`.  Enable pinn repo to run on cluster:

1. Install miniconda https://docs.conda.io/en/latest/miniconda.html :

    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
2. Create the virtual environment using `environment.yml` which includes all necessary packages. 
    ```bash
    conda env create -f environment.yml
    ```

3. Activate the generated venv (`pinn-env`) 
    ```bash
    conda activate pinn-env
    ```
4. To test cluster, submit a job on a compute node. This is achieved through `test_cluster.sh`. 
    ```bash
    $ sbatch $HOME/pinnswithdxde/tests/integration_tests/cluster/test_cluster.sh
    ```
> `NOTE`: Do not forget to adopt the inside of the `test_cluster.sh` to specify the slurm options e.g., `--mail-user`.

> `NOTE`: Always be sure that you activated venv `pinn-env` (step 3) before `sbatch` any slurm sript. This includes other scripts you will run as well. The reason behind is that activating venv in `test_cluster.sh` needs the full path for the conda env `pinn-env` and it gives some **init** error if the full path is used.  

> `NOTE`: For conda commands: A conda [cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) can be very useful. 