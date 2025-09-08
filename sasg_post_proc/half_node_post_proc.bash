#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --time=04:00:00
#SBATCH --mem=20GB
#SBATCH --account=project_2007848
#SBATCH --partition=small 
#SBATCH -o ./slurm_out_post_proc.txt

source /scratch/project_2007848/enchanted_venv/bin/activate

python3 -u sasg_post_proc/sasg_post_proc.py $1 > out.txt