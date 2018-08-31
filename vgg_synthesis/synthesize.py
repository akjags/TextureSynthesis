import numpy as np
import tensorflow as tf
import os
import sys

import TextureSynthesis as ts
from VGGWeights import *
from model import *

def main(model_name, texture, saveDir, nSpl):
    # Load VGG-19 weights and build model
    vgg_weights = VGGWeights('vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()

    # Load tensorflow session
    sess = tf.Session()

    # Weights for each layer
    pool5_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9, "pool3": 1e9, "pool4": 1e9, "pool5": 1e9}
    pool4_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9, "pool3": 1e9, "pool4": 1e9}
    pool3_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9, "pool3": 1e9}
    pool2_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9}
    pool1_weights = {'conv1_1': 1e9, 'pool1': 1e9}
    layer_weights = {'pool1': pool1_weights, 'pool2': pool2_weights, 'pool3': pool3_weights, 'pool4': pool4_weights, 'pool5': pool5_weights}

    this_layer_weight = layer_weights[model_name]

    textures_directory = "orig_ims"

    # Set number of iterations
    iterations = 10001

    print "Synthesizing texture", texture, "matching model", model_name, "for", iterations, "iterations, with nSplits:", nSpl
    image_name = texture.split(".")[0]
    filename = textures_directory + "/" + texture
    img = np.load(filename)

    # Initialize texture synthesis
    text_synth = ts.TextureSynthesis(sess, my_model, img, this_layer_weight, model_name, image_name, saveDir, iterations, nSpl)

    # Do training
    text_synth.train()
    sys.stdout.flush()

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        model_name = 'pool4'
        texture = 'tulips.npy'
        saveDir = 'v1'
        nSpl = 2
    else:
        model_name = args[1]
        texture = args[2] + '.npy'
        saveDir = args[3]
        nSpl = float(args[4])
    main(model_name, texture, saveDir, nSpl)

