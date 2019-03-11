#!/bin/sh

base_stim_dir="$HOME/TextureSynthesis/stimuli/grant"
orig_im_dir="$base_stim_dir/orig_bw"
save_dir="$base_stim_dir/tex_bw"

for filename in $orig_im_dir/*.jpg;
do
  a=${filename##*/};
  img=${a%.*};
  for layer in "pool1" "pool2" "pool3" "pool4";
  do
    for sample in `seq 1 10`;
    do
      for nPools in "1" "2" "3" "4";
      do
        echo $img $layer $nPools $sample
        sbatch -p hns,gpu --gres gpu:1 --mem=5000 --time=00:60:00 --wrap="module load py-tensorflow; module load py-scikit-image; module load py-scipystack; cd $HOME/TextureSynthesis; python synthesis_scripts/preprocess_image_jpg.py $orig_im_dir $img; python vgg_synthesis/synthesize.py $layer $orig_im_dir $img $save_dir/s$sample $nPools"
        sleep 1
      done
    done
  done
done
