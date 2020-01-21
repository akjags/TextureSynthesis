import numpy as np
import argparse
from scipy.misc import imresize
import tensorflow as tf
import sys
sys.path.insert(0, '/home/users/akshayj/TextureSynthesis/tensorflow_synthesis')
from ImageUtils import *
from model import *
from TextureSynthesis import *

def get_gram_mtx(filepath, nSplits, get_activations = 0):
    # Use file path to get name of image and path.
    filename = filepath.split('/')[-1]
    tex_dir = '/'.join(filepath.split('/')[:-1])

    # Setup variables to pass into the TextureSynthesis function.
    weights = {'conv1_1': 1e9, 'pool1': 1e9, 'pool2': 1e9, 'pool3': 1e9, 'pool4': 1e9}
    vgg_weights = VGGWeights('/home/users/akshayj/TextureSynthesis/tensorflow_synthesis/vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()

    print 'Loading image: %s' % (tex_dir + '/' + filename)
    img = np.load(tex_dir + '/' + filename)
    
    if len(img.shape) < 4:
        img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])

    if img.shape[1] < 256:
        img = imresize(img[0], (256,256))
        img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])

    with tf.Session() as sess:
        ts = TextureSynthesis(sess, my_model, img, weights, 'pool4', filename.split('.')[0], '', 10001, nSplits)

    if get_activations == 1:
        output = ts._get_activations()
    else: # gramian
        output  = ts.constraints
    tf.reset_default_graph()
    return output
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--filepath", default="/scratch/groups/jlg/texture_stimuli/color/textures")
    parser.add_argument("-s", "--saveDir", default="/scratch/groups/jlg/texture_stimuli/color/gram_features")
    parser.add_argument("-n", "--nSplits", default=1, type=int)
    parser.add_argument("-a", "--getActivations", default=0, type=int)
    args = parser.parse_args()
    # Get inputs

    if args.getActivations==1:
        savetype='activation'
    else:
        savetype='gram'

    print('Computing {} for {} with {} x {}'.format(savetype, args.filepath, args.nSplits, args.nSplits))
    filename = args.filepath.split('/')[-1]

    gram_mtx = get_gram_mtx(args.filepath, args.nSplits, args.getActivations)
    savePath = '{}/{}{}x{}_{}'.format(args.saveDir, savetype, args.nSplits, args.nSplits, '.'.join(filename.split('.')[:-1]))
    print('Saving to {}'.format(savePath))
    np.save(savePath, gram_mtx)
