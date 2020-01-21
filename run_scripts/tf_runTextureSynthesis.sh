#!/bin/sh

base_stim_dir="$PI_SCRATCH/texture_stimuli/bw"
orig_im_dir="$base_stim_dir/originals"
save_dir="$base_stim_dir/textures"
#nSamples=4

#for filename in $orig_im_dir/*.{png,jpg};
#  a=${filename##*/};
#  img=#{a%.*};
for img in "phasescrambledcrystal" "phasescrambledfur";
do
  for layer in "pool1" "pool2" "pool4";
  do
    for sample in 1 2 3 4; 
    do
      for nPools in "1" "2" "3" "4";
      do
        fnm="$save_dir/${nPools}x${nPools}_${layer}_${img}_smp${sample}.npy"

        if [[ ! -f $fnm ]]
        then
          echo $fnm "not found"
          #sbatch -p hns,gpu --gres gpu:1 --mem=5000 --time=02:00:00 --wrap="module load py-tensorflow; module load py-scikit-image/0.15.0_py27; module load py-scikit-learn; module load py-scipystack; cd $HOME/TextureSynthesis; python tensorflow_synthesis/synthesize.py -l $layer -d $orig_im_dir -i $img -o $save_dir -s $sample -p $nPools"
          python tensorflow_synthesis/synthesize.py -l $layer -d $orig_im_dir -i $img -o $save_dir -s $sample -p $nPools;
          #sleep 0.2
        fi
      done
    done
  done
done
