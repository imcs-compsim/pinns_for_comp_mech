import os
import pytest
import sys
from pathlib import Path
import subprocess

# Which input files will be used 
input_files =   [
                "Euler_beam_cantiliver_static.py",
                "Euler_beam_fixed_static.py",
                "Euler_beam_simply_complex_static.py",
                "Euler_beam_simply_dynamic.py",
                "Euler_beam_simply_point_static.py",
                "Euler_beam_simply_static.py"
                ]

# test function
def test_beams(pinn_input_file):
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
   
    base_beam_file = dirpath.parent.parent.parent.joinpath("beams").joinpath(example_file_name)
    test_beam_file = tmpdir.strpath +"/" + example_file_name
    
    file_test = open(test_beam_file,'w')
    file_base = open(base_beam_file,'r')

    for line in file_base:
        file_test.write(line)
        if "model = dde.Model(data, net)" in line:
            compile_line = 'model.compile("adam", lr=0.0001)\n'
            file_test.write(compile_line)
            train_line = 'model.train(epochs=1)\n'
            file_test.write(train_line)
            file_test.close()
            break
    
    return test_beam_file