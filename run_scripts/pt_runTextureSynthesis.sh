#!/bin/sh

base_stim_dir="$HOME/TextureSynthesis/stimuli/tex-fMRI"
orig_im_dir="orig"
save_dir="tex2"

for filename in $base_stim_dir/$orig_im_dir/*.png;
do
  a=${filename##*/};
  img=${a%.*};
  for layer in "pool2" "pool4";
  do
    for sample in `seq 1 12`;
    do
      for nPools in "1";
      do
        fnm="$base_stim_dir/$save_dir/s$sample/${nPools}x${nPools}_${layer}_${img}.npy"

        if [[ ! -f $fnm ]]
        then
          echo $fnm "not found"
          sbatch -p hns,gpu --gres gpu:1 --mem=10G --time=00:15:00 --wrap="module load py-scikit-image/0.15.0_py27; module load py-pytorch/1.0.0_py27; module load py-scikit-learn/0.19.1_py27; module load py-scipystack; cd $HOME/TextureSynthesis; python synthesis_scripts/preprocess_image.py $orig_im_dir $img; python pytorch_synthesis/pt_synthesize.py -d $base_stim_dir -i \"$orig_im_dir/${img}.png\" -o \"$save_dir/s$sample\" -l $layer"
          sleep 0.5
        fi
      done
    done
  done
done
