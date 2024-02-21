#!/bin/bash
#SBATCH --job-name=hamed_server
#SBATCH --gres=gpu:v100l:3
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH -c 24
#SBATCH --mem=52000
#SBATCH --job-name=test2_jupyter
#SBATCH --output=notebook_output.log

echo Running on $(hostname)

# activate environment
module load gcc python arrow
source /home/hamedth/projects/def-hemmati-ac/hamedth/venv3.9/bin/activate


# choose a random port
jupyter lab --ip 0.0.0.0 --port 2244
