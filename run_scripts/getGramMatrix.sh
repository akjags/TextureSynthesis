#!/bin/sh

cnt=0
tex_dir="$PI_SCRATCH/tex_eq"
save_dir="$PI_SCRATCH/gram_texOB"
mkdir $save_dir

for nPools in "1"
do
  for subdir in {1..10}
  do
    for filepath in $tex_dir/s$subdir/*.npy;
    do

      fnm=$(basename $filepath)
      name="${fnm%.*}"
      gfnm="gram${nPools}x${nPools}_${name}_smp${subdir}.npy"
      if [[ ! -f $save_dir/$gfnm ]]
      then
        echo "Computing $nPools x $nPools gram matrix of $fnm sample: $subdir" 
        sbatch -p hns,gpu --gres gpu:1 --mem=5000 --time=00:10:00 --wrap="module load py-tensorflow; module load py-scipystack; cd $HOME/TextureSynthesis; python analysis_scripts/get_gram_mtx.py $filepath $save_dir $nPools $subdir"
        #sleep 0.25
        #python analysis_scripts/get_gram_mtx.py $filepath $save_dir $nPools $subdir
        cnt=$((cnt+1))
      fi
    done
  done
done

echo $cnt

