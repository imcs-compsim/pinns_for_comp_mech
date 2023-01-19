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

pytest $HOME/pinnswithdxde/tests/integration/tests/test_cluster.py

echo
echo "Job finished with exit code $? at: `date`"