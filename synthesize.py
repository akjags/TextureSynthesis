import numpy as np
import tensorflow as tf
import os
import sys

import TextureSynthesis as ts
from VGGWeights import *
from model import *

def main(model_name, texture):
    # Load VGG-19 weights and build model
    vgg_weights = VGGWeights('vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()

    # Load tensorflow session
    sess = tf.Session()

    # Weights for each layer
    pool4_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9, "pool3": 1e9, "pool4": 1e9}
    pool3_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9, "pool3": 1e9}
    pool2_weights = {"conv1_1": 1e9, "pool1": 1e9, "pool2": 1e9}
    pool1_weights = {'conv1_1': 1e9, 'pool1': 1e9}
    layer_weights = {'pool1': pool1_weights, 'pool2': pool2_weights, 'pool3': pool3_weights, 'pool4': pool4_weights}

    this_layer_weight = layer_weights[model_name]

    textures_directory = "orig_ims"

    print "Synthesizing texture", texture, "matching model", model_name
    image_name = texture.split(".")[0]
    filename = textures_directory + "/" + texture
    img = np.load(filename)

    # Initialize texture synthesis
    text_synth = ts.TextureSynthesis(sess, my_model, img, this_layer_weight, model_name, image_name)

    # Do training
    text_synth.train()
    sys.stdout.flush()

if __name__ == "__main__":
    model_name = 'pool4'
    texture = 'tulips.npy'
    main(model_name, texture)

