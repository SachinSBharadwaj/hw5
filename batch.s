#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1 
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=jacobi
#SBATCH --mail-type=END 
#SBATCH --mail-user=ssb638@nyu.edu
#SBATCH --output=output_jacobi.out

mpirun mpi_jacobi 400 10000 100

