# Texture Synthesis
This repository contains the code, figures, and notebooks used for synthesizing the textures used as stimuli in behavioral and neuroimaging experiments.

Contact: akshayj at stanford dot edu

Usage:
    
    cd TextureSynthesis/tensorflow_synthesis # cd into the tensorflow_synthesis directory before running.

    POOLSIZE="2" # "1" for 1x1, "2" for 2x2, "3" for 3x3 or "4" for 4x4.
    LAYER="pool4" # "pool1", "pool2", "pool3", "pool4", "pool5"
    NUM_SAMPLES_TO_GENERATE=3

    python synthesize.py -i /path/to/original_image.png -o /path/to/output/dir -p POOLSIZE -l LAYER -g 1 -s NUM_SAMPLES_TO_GENERATE
