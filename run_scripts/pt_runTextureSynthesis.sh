#!/bin/sh

base_stim_dir="$HOME/TextureSynthesis/stimuli/textures"
orig_im_dir="$base_stim_dir/orig_color"
save_dir="$base_stim_dir/out_color"

for filename in $orig_im_dir/*.{png,jpg};
do
  a=${filename##*/};
  img=${a%.*};
  for layer in "pool1" "pool2" "pool3" "pool4";
  do
    for sample in `seq 1 10`;
    do
      for nPools in "1";
      do
        fnm="$save_dir/s$sample/${nPools}x${nPools}_${layer}_${img}.npy"

        if [[ ! -f $fnm ]]
        then
          echo $fnm "not found"
          sbatch -p hns,gpu --gres gpu:1 --mem=5000 --time=00:30:00 --wrap="module load py-scikit-image; module load py-pytorch/1.0.0_py27; module load py-scipystack; cd $HOME/TextureSynthesis; python synthesis_scripts/preprocess_image_jpg.py $orig_im_dir $img; python pytorch_synthesis/pt_synthesize.py -i \"$orig_im_dir/${img}.png\" -o \"$save_dir/s$sample\" -l $layer"
          sleep 0.5
        fi
      done
    done
  done
done
