#!/bin/sh

for IMG in "bananas" "bark" "bison" "blossoms" "blotch" "braids" "bricks" "bubbly" "bumpy" "buns" "crystals" "dalmatians" "ducks" "face" "frills" "fur" "galaxy" "gourds" "grass" "honeycomb" "lace" "marbled" "marbles" "monarchs" "paisley" "pears" "phlox" "rorschach" "spiky" "splotchy" "stars" "succulent" "tiles";
do
  sbatch -p hns --gres gpu:1 --mem=5000 --time=00:30:00 --wrap="module load py-tensorflow; module load py-scipystack; cd $HOME/TextureSynthesis; python TextureSynthesis.py $IMG"
        sleep 1
done
