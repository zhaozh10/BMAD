#!/bin/bash
#SBATCH --job-name=bmad               # Job name
#SBATCH --output=slurm-%j.out            # Output file (%j = Job ID)
#SBATCH --error=slurm-%j.err             # Error file
#SBATCH --partition=defq                 # Partition name (defq or L20)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --gres=gpu:1                     # Number of GPUs (adjust as needed)
#SBATCH --time=40:00:00                  # Walltime (1 hour)
#SBATCH --cpus-per-task=8                # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=64G                        # RAM per node (adjust as needed)


nvidia-smi

source ~/.bashrc
conda init
# Load your environment, e.g. Anaconda or modules
conda activate csflow  # <-- Replace with your conda env name

python main.py --mode train --data chest --model cutpaste

python main.py --mode test --data chest --model cutpaste --weight results/csflow/chest/run/weights/model.ckpt


# sleep infinity  # Keep the job alive to check results