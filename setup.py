import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="pinnswithdxde",
    version="0.1",
    author="Tarik Sahin, Max von Danwitz",
    author_email="tarik.sahin@unibw.de, max.danwitz@unibw.de",
    description=("A library for PINNs on Computational Mechanics"),
    keywords="PINNs, Computational Mechanics, Hyperparameter optimization",
    packages=[
        'utils',
        'utils.contact_mech',
        'utils.deep_energy',
        'utils.elasticity',
        'utils.geometry',
        'utils.hyperelasticity',
        'utils.postprocess'
    ],

    long_description=read('README.md'),
    tests_require='pytest',
)