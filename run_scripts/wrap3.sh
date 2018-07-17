#!/bin/sh

for IMG in "biryani" "bubbles" "cherries" "clouds" "crowd" "dahlias" "fireworks" "leaves" "noodles" "rocks" "tulips" "worms" "zebras" "bananas" "bark" "bison" "blossoms" "blotch" "braids" "bricks" "bubbly" "bumpy" "buns" "crystals" "dalmatians" "ducks" "face" "frills" "fur" "galaxy" "gourds" "grass" "honeycomb" "lace" "marbled" "marbles" "monarchs" "paisley" "pears" "phlox" "rorschach" "spiky" "splotchy" "stars" "succulent" "tiles";
do
  for LAYER in "pool1" "pool2" "pool4";
  do
      for OBSRF in "1x1" "2x2" "3x3" "4x4";
      do
        for IMRF in "1x1" "2x2" "3x3" "4x4";
        do
          sbatch -p hns,gpu --gres gpu:1 --mem=5000 --time=00:60:00 --wrap="module load py-tensorflow; module load py-scipystack; cd $HOME/TextureSynthesis; python get_gram_mtx.py $IMG $LAYER $IMRF $OBSRF"
          sleep 1
      done
    done
  done
done
