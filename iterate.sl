#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64g
#SBATCH -t 1-

module restore

python ea_analysis.py
