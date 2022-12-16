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
For cluster, we concluded that we should use `conda` since we had issues in terms of package installation particularly the package `gmsh`.  Enable pinn repo to run on cluster:

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
4. Test cluster (check the folder below to see which tests, examples will be run)
    ```bash
    $ pytest tests/integration_tests/cluster
    ```
