import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, './vgg_synthesis')
from ImageUtils import *
from model import *
from TextureSynthesis import *

if __name__ == "__main__":
    args = sys.argv

    # Get inputs
    filepath = args[1]
    saveDir = args[2]
    nSplits = float(args[3])
    smpIdx = float(args[4])

    # Use file path to get name of image and path.
    filename = filepath.split('/')[-1]
    tex_dir = '/'.join(filepath.split('/')[:-1])

    # Setup variables to pass into the TextureSynthesis function.
    weights = {'conv1_1': 1e9, 'pool1': 1e9, 'pool2': 1e9, 'pool3': 1e9, 'pool4': 1e9}
    vgg_weights = VGGWeights('vgg_synthesis/vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()

    # This is where you specify how many splits you want
    nSplits = 1 # Just get 1x1 splits

    sess = tf.Session()

    print 'Loading image: %s' %(tex_dir + '/' + filename)
    img = np.load(tex_dir + '/' + filename)
    
    if len(img.shape) < 4:
        img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])

    ts = TextureSynthesis(sess, my_model, img, weights, 'pool4', filename.split('.')[0], saveDir, 10001, nSplits)
    np.save('%s/gram%gx%g_%s_smp%g' % (saveDir, nSplits,nSplits,filename.split('.')[0], smpIdx), ts.constraints)    
