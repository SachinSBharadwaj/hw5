#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1 
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=samsort
#SBATCH --mail-type=END 
#SBATCH --mail-user=ssb638@nyu.edu
#SBATCH --output=sort1.out

mpirun ssort1 

