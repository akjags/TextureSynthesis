#!/bin/bash 
#
#SBATCH --job-name=tulips1
#SBATCH --output=/home/akshayj/log/tulips1.%j.out
#SBATCH --error=/home/akshayj/log/tulips1.%j.err
#SBATCH --time=00:30:00
#SBATCH --gres gpu:1
#SBATCH -p gpu,hns_gpu
#SBATCH --nodes=1
#SBATCH --mem=4000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=akshayj@stanford.edu

module load tensorflow/0.9.0
module load python/2.7.5

cd /home/akshayj/TextureSynthesis
mkdir output
srun  python ./synthesize.py 'pool1' 'tulips'
