#!/usr/bin/bash

#SBATCH --job-name=auto_car
#SBATCH --output=./output/auto_car.%j.out
#SBATCH --error=./output/auto_car.%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB

module load python/3.11.7
module load py-pip/23.1.2_python/3.11.7

python3 -m venv myenv
source myenv/bin/activate

pip install -r requirements.txt

python curriculum.py
