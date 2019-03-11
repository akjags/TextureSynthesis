#!/bin/sh

cnt=0
tex_dir="$PI_SCRATCH/grant/orig_bw"
save_dir="$PI_SCRATCH/grant/gram_mtx_orig_bw"
mkdir $save_dir

for nPools in "1" "2" "3" "4"
do
    for filepath in $tex_dir/*.npy;
    do
      fnm=$(basename $filepath)
      gfnm="gram${nPools}x${nPools}_${fnm}"
      if [[ ! -f $save_dir/$gfnm ]]
      then
        echo "Computing gram matrix of $fnm"
        sbatch -p hns,gpu --gres gpu:1 --mem=5000 --time=00:10:00 --wrap="module load py-tensorflow; module load py-scipystack; cd $HOME/TextureSynthesis; python analysis_scripts/get_gram_mtx.py $filepath $save_dir $nPools"
        sleep 0.25
      else
        echo "Gram matrix already exists: $fnm"
      fi
      cnt=$((cnt+1))
    done
  done

echo $cnt

