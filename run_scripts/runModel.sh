#!/bin/sh

sbatch -p hns,normal --mem=5000 --time=04:00:00 --wrap="module load py-scikit-learn; module load py-scipystack; cd $HOME/TextureSynthesis; python analyzeGramBehavior.py"
