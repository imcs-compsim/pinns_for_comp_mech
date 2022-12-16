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
