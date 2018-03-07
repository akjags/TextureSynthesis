#!/bin/bash 
#
#SBATCH --job-name=txsh
#SBATCH --output=/home/akshayj/log/txsh.%j.out
#SBATCH --error=/home/akshayj/log/txsh.%j.err
#SBATCH --time=00:35:00
#SBATCH --gres gpu:1
#SBATCH -p gpu,hns_gpu
#SBATCH --nodes=1
#SBATCH --mem=2000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=akshayj@stanford.edu

module load tensorflow/0.9.0
module load python/2.7.5

cd /home/akshayj/TextureSynthesis

vname='v1'
mkdir vname
srun python ./synthesize.py 'pool5' 'fronds' $vname
