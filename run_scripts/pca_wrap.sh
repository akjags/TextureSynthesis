#!/bin/bash

out_dir="$HOME/TextureSynthesis/stimuli/textures/nmf"

for input_img in "orig_bw/rocks.jpg" "orig_bw/glass.jpg" "orig_bw/drops.jpg" "orig_color/tulips.jpg" "orig_color/paisley.jpg" "orig_bw/bark.jpg" "orig_bw/bricks.jpg" "orig_color/fireworks.jpg" "orig_color/face.jpg" "orig_color/fronds.jpg"
do
  for sample in `seq 1 4`
  do
    for which_pc in `seq 0 5`
    do
      c=$(echo ${input_img##*/} | cut -d'.' -f1)
      fnm="$out_dir/v$sample/1x1_pool2_${c}_NMF${which_pc}_10.png"
      if [[ ! -f $fnm ]]
      then
        echo $fnm "not found"
        sbatch -p hns,gpu --gres gpu:1 --time=00:60:00 --wrap="cd ~/TextureSynthesis/run_scripts; bash pca_texSynth.sh $input_img $sample $which_pc $out_dir"
        sleep 0.25
      fi
    done
  done
done

