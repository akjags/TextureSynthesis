#!/bin/sh

module load py-scikit-image
module load py-pytorch/1.0.0_py27
module load py-scikit-learn
module load py-scipystack

inPath=$1
sample=$2
which_pc=$3
out_dir=$4

cd ~/TextureSynthesis/pytorch_synthesis

for interval in -10 -5 -2 0 2 5 10;
do
  c=$(echo ${inPath##*/} | cut -d'.' -f1)
  fnm="$out_dir/v$sample/1x1_pool2_${c}_NMF${which_pc}_${interval}.png"

  if [[ ! -f $fnm ]]
  then
    echo "Generating file: " $fnm
    python pt_synthesize.py -o "nmf/v$sample" -i "$inPath" -g "nmf" -z $interval -p $which_pc
  fi
done

