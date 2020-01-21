#!/bin/sh

cnt=0
tex_dir="$PI_SCRATCH/texture_stimuli/color/textures"
save_dir="$PI_SCRATCH/texture_stimuli/color/gram_features"
mkdir $save_dir

for nPools in 1 2 3 4 5 6
do
    for filepath in $tex_dir/*.npy;
    do
      fnm=$(basename $filepath)
      name="${fnm%.*}"
      gfnm="gram${nPools}x${nPools}_${name}.npy"

        if [[ ! -f $save_dir/$gfnm ]]
        then
          echo "Computing $nPools x $nPools gram matrix of $fnm sample: $subdir" 
          python analysis_scripts/get_gram_mtx.py -p $filepath -s $save_dir -n $nPools
          cnt=$((cnt+1))
        else
          echo "$gfnm already exists; skipping..."
        fi
    done
done
echo $cnt

