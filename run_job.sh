#!/usr/bin/sh

#SBATCH --job-name=birdie_job
#SBATCH --output=/projects/ashehu/amoldwin/logs/birdie_job-%j.output
#SBATCH --error=/projects/ashehu/amoldwin/logs/birdie_job-%j.error
#SBATCH --mail-user=<amoldwin@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=gpuq  
#SBATCH --qos=gpu # gpu, cs_dept
#SBATCH --nodes=1 
##SBATCH --ntasks-per-node=64 # I cannot do this,
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --mem=128G 
#SBATCH --time=1-00:00:00


source ~/envs/mambaenv/bin/activate

python train_tiny_model.py