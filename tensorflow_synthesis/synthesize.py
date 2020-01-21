import numpy as np
import tensorflow as tf
import os
import sys, time
import argparse

import TextureSynthesis as ts
from VGGWeights import *
from model import *
from skimage.io import imread, imsave

def main(args):
    # Keep track of how long this all takes
    start_time = time.time()

    # Load VGG-19 weights and build model
    vgg_weights = VGGWeights('tensorflow_synthesis/vgg19_normalized.pkl')
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
    this_layer_weight = layer_weights[args.layer]

    # Make a temporary directory
    tmpDir = "%s/iters" % (args.outputdir)
    os.system("mkdir -p %s" %(tmpDir))

    print "Synthesizing texture", args.image, "matching model", args.layer, "for", args.iterations, "iterations, with nPools:", args.nPools
    img = np.load('{}/{}.npy'.format(args.inputdir, args.image))

    # Initialize texture synthesis
    text_synth = ts.TextureSynthesis(sess, my_model, img, this_layer_weight, args.layer, args.image, tmpDir, args.iterations+1, args.nPools)

    # Do training
    if args.generateMultiple==1:
        for i in range(args.sampleidx):
            print('Generating sample {} of {}'.format(i+1, args.sampleidx))
            text_synth.train(i+1)
    else:
        text_synth.train(args.sampleidx) 
    postprocess_img(tmpDir, args)

    print('DONE. This took {} seconds'.format(time.time()-start_time))
    sys.stdout.flush()

def postprocess_img(raw, args):
    for im in os.listdir(raw):
        if 'step_{}.npy'.format(args.iterations) in im and '{}x{}_{}_{}'.format(args.nPools, args.nPools, args.layer, args.image) in im:
            imName = raw+'/'+im
            imi = np.load(imName)
            outName = '{}/{}'.format(args.outputdir, im[:im.index('_step')])

            # Save as PNG into outdir
            imsave(outName + '.png', imi)

            # Also copy .npy fileinto outdir
            os.system('cp %s %s.npy' % (imName, outName))
            print im + ' saved as PNG to ' + args.outputdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer", default="pool2")
    parser.add_argument("-d", "--inputdir", default="/scratch/groups/jlg/texture_stimuli/color/originals")
    parser.add_argument("-o", "--outputdir", default="/scratch/groups/jlg/texture_stimuli/color/textures")
    parser.add_argument("-i", "--image", default="rocks")
    parser.add_argument("-s", "--sampleidx", type=int, default=1)
    parser.add_argument("-p", "--nPools", type=int, default=1)
    parser.add_argument('-g', '--generateMultiple',type=int, default=0)
    parser.add_argument('-n', '--iterations', type=int, default=10000)
    args = parser.parse_args()
    main(args)
    #tmpDir = "%s/iters" % (args.outputdir)
    #postprocess_img(tmpDir, args)
 
