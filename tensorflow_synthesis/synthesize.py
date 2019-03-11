import numpy as np
import tensorflow as tf
import os
import sys

import TextureSynthesis as ts
from VGGWeights import *
from model import *
from skimage.io import imread, imsave

def main(layer_name, orig_im_dir, imgName, saveDir, nPools):
    # Load VGG-19 weights and build model
    vgg_weights = VGGWeights('vgg_synthesis/vgg19_normalized.pkl')
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

    this_layer_weight = layer_weights[layer_name]

    # Make a temporary directory
    tmpDir = "%s/iters" % (saveDir)
    os.system("mkdir -p %s" %(tmpDir))

    # Set number of iterations
    iterations = 10001

    print "Synthesizing texture", imgName, "matching model", layer_name, "for", iterations, "iterations, with nPools:", nPools
    image_name = imgName.split(".")[0]
    filename = orig_im_dir + "/" + imgName
    img = np.load(filename)

    # Initialize texture synthesis
    text_synth = ts.TextureSynthesis(sess, my_model, img, this_layer_weight, layer_name, image_name, tmpDir, iterations, nPools)

    # Do training
    text_synth.train()

    
    postprocess_img(tmpDir, saveDir, image_name, layer_name);

    sys.stdout.flush()

def postprocess_img(raw, out, img_name, layer_name):
    for im in os.listdir(raw):
        if 'step_10000.npy' in im and img_name in im and layer_name in im :
            imName = raw+'/'+im
            imi = np.load(imName)
            outName = out+'/'+im[:im.index('_step')]

            # Save as PNG into outdir
            imsave(outName + '.png', imi)

            # Also copy .npy fileinto outdir
            os.system('cp %s %s.npy' % (imName, outName))
            print im + ' saved as PNG to ' + out

if __name__ == "__main__":
    args = sys.argv
    layer_name = args[1]
    orig_im_dir = args[2]
    imgName = args[3] + '.npy'
    saveDir = args[4]
    nPools = float(args[5])

    main(layer_name, orig_im_dir, imgName, saveDir, nPools)

