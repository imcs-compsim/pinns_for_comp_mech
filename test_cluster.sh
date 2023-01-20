
#!/bin/sh
echo '
###############################################
#             PINNs on Cluster                #
###############################################
'
# Define the user mail address if 
##SBATCH --mail-user=<name>.<surname>@unibw.de
##SBATCH --mail-type=BEGIN,END,FAIL,NONE


# activate the conda environment
$HOME/anaconda3/bin/conda activate pinn-env
# the run cluster test
#pytest tests/integration_tests/cluster