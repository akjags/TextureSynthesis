#!/bin/sh

for IMG in "ramen";
do
  for LAYER in "pool1" "pool2" "pool3" "pool4";
  do
    for SAVEDIR in "s1" "s2" "s3";
    do
      for NSPL in "2" "3" "4";
      do
        sbatch -p hns --gres gpu:1 --mem=5000 --time=00:60:00 --wrap="module load py-tensorflow; module load py-scipystack; cd $HOME/TextureSynthesis; python ./synthesize.py $LAYER $IMG $SAVEDIR $NSPL"
        sleep 1
      done
    done
  done
done
