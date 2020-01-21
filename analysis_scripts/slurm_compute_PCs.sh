#!/bin/bash

module load py-scikit-image/0.15.0_py27
module load py-pytorch/1.0.0_py27
module load py-scikit-learn
module load py-scipy/1.1.0_py27
module load py-numpy/1.14.3_py27
module load py-matplotlib/2.2.2_py27


module load py-tensorflow/1.6.0_py27

cd ~/TextureSynthesis/analysis_scripts

python compute_PCs.py
