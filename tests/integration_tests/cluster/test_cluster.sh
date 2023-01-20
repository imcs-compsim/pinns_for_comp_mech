#!/bin/bash
##########################################
#                                        #
#  Specify your SLURM directives         #
#                                        #
##########################################
#
# User's Mail:
##SBATCH --mail-user=<name>.<surname>@unibw.de
#
# When to send mail?:
##SBATCH --mail-type=BEGIN,END,FAIL
#
# Job name:
#SBATCH --job-name=PINN_cluster_test
#
# Output file:
#SBATCH --output=slurm-%j-%x.out
#
# Error file:
#SBATCH --error=slurm-%j-%x.err
#
# Standard case: specify only number of cpus
#SBATCH --ntasks=8
#
# Walltime: (days-hours:minutes:seconds)
#SBATCH --time=00:10:00
#
##########################################
#                                        #
#  Advanced SLURM settings	          #
#                                        #
##########################################
#
# If you want to specify a certain number of nodes:
##SBATCH --nodes=1
#
# and exactly 'ntasks-per-node' cpus on each node:
##SBATCH --ntasks-per-node=8
#
# Allocate full node and block for other jobs:
#SBATCH --exclusive
#
# Request specific hardware features:
##SBATCH --constraint="skylake|cascadelake"
#
###########################################

# Setup shell environment and start from home dir
# 
echo $HOME
cd $HOME

export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=$OMP_NUM_THREADS

echo "Job starts at: `date`"

# measure the time
SECONDS=0 
python $HOME/pinnswithdxde/tests/integration_tests/cluster/tf_cluster_settings.py
pytest $HOME/pinnswithdxde/tests/integration_tests/cluster/test_cluster.py
duration=$SECONDS

echo
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
echo "Number of threads: $OMP_NUM_THREADS"

echo
echo "Job finished with exit code $? at: `date`"