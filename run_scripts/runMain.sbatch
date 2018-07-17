#!/bin/bash 
#
#SBATCH --job-name=4p4bub
#SBATCH --time=00:60:00
#SBATCH -p hns,gpu
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mem=5000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=akshayj@stanford.edu

if [[ "$SHERLOCK" == "1" ]]; then
  module load tensorflow/0.9.0
  module load python/2.7.5
  echo "Running on Sherlock 1.0"

elif [[ "$SHERLOCK" == "2" ]]; then
  module load py-tensorflow
  module load py-scipystack
  echo "Running on Sherlock 2.0"
fi

cd $HOME/TextureSynthesis
savedir='s3'
layer='pool4'
im='bubbles'
nSpl='4'
mkdir $savedir
srun python ./synthesize.py $layer $im $savedir $nSpl
