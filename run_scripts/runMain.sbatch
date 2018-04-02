#!/bin/bash 
#
#SBATCH --job-name=p4ns2
#SBATCH --time=00:30:00
#SBATCH -p gpu,hns
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mem=2000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=akshayj@stanford.edu

if [[ "$SHERLOCK" == "1" ]]; then
  module load tensorflow/0.9.0
  module load python/2.7.5
  echo "Running on Sherlock 1.0"

elif [[ "$SHERLOCK" == "2" ]]; then
  module load py-tensorflow/1.3.0_py27
  module load py-scipystack/1.0_py27
  module load python/2.7.13
  echo "Running on Sherlock 2.0"
fi

cd $HOME/TextureSynthesis
vname='texture_output'
nSpl='2'
layer='pool4'
im='forest'
mkdir $vname
srun python ./synthesize.py $layer $im $vname $nSpl
