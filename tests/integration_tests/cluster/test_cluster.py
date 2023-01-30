import os
import pytest
import sys
from pathlib import Path
import subprocess

# Which input files will be used
# It would make sense to test only one example from each folder: 
# - Euler_beam_cantiliver_static from "beams"
# - Lame_problem_quarter_gmsh from "elasticity_2d"
# - heatEq2D from "heatEquation"

input_files =   [
                "beams/Euler_beam_cantiliver_static.py",
                "elasticity_2d/linear_elasticity/lame/Lame_problem_quarter_gmsh.py",
                "heatEquation/heatEq2D.py"
                ]

# test pinn model on cluster
def test_cluster(pinn_input_file):
    print(pinn_input_file)
    # run the input file using the current python interpreter
    subprocess.run([sys.executable,pinn_input_file])

# take an input file, read until model is generated, 
# add lines for compiling with "adam" and traininig it for only 1 epoch,
# save the results in test file 
@pytest.fixture(params=input_files)
def pinn_input_file(tmpdir,request):
    """ 
    Creates the input file for each model
    """
    dirpath = Path(os.path.dirname(__file__))
    example_file_name = request.param
   
    base_model_file = dirpath.parent.parent.parent.joinpath(example_file_name)
    test_model_file = tmpdir.strpath +"/" + example_file_name.split("/")[1]
    
    file_base = open(base_model_file,'r')
    file_test = open(test_model_file,'w')
    
    for line in file_base:
        file_test.write(line)
        if "model = dde.Model(data, net)" in line:
            compile_line = 'model.compile("adam", lr=0.0001)\n'
            file_test.write(compile_line)
            train_line = 'model.train(epochs=1)\n'
            file_test.write(train_line)
            file_test.close()
            break
    
    return test_model_file