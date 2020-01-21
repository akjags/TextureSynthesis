from __future__ import print_function
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import copy, os, time, glob
import argparse, pdb

from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from get_gram_mtx import get_gram_mtx

###
def get_feature_maps(thisLayer, thisRF, tex_dir, save_feature_maps=1, save_dir='.'):
   # Specify which layers to match
  all_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4']
  this_layers = all_layers[:all_layers.index(thisLayer)+1]

  #
  nSplits = int(thisRF.split('x')[0])

  tStart = time.time()
  # Get the feature maps for each image in the tex_dir
  all_imgs = glob.glob('{}/*.npy'.format(tex_dir))
  labels = []; feature_maps = [];
  for i in range(len(all_imgs)):
    thisTex = all_imgs[i]

    gram_mtx = get_gram_mtx(thisTex, nSplits)

    # Extract gram matrix at this layer and unravel it into a vector.
    layer_features = gram_mtx[thisLayer].ravel()

    # get the label and feature map
    feature_maps.append(layer_features)
    label = thisTex.split('/')[-1].split('_')[0]
    # labels.append(label)

    print('----Img {} of {}: {} - Shape={} - Label={}----'.format(i, len(all_imgs), thisTex, layer_features.shape, label))

  tElapsed = time.time() - tStart
  print('Extracting {} feature maps for {} images took {} seconds'.format(thisLayer, len(all_imgs), tElapsed))
  feature_maps = np.stack(feature_maps, axis=-1)

  #feats = {}
  #feats['labels'] = labels
  #feats['features'] = feature_maps
  if save_feature_maps:
    print('Saving feature maps of size: {}'.format(feature_maps.shape))
    np.save('{}/{}_{}_DTD_features.npy'.format(save_dir, thisRF, thisLayer), feature_maps)
    print('Saved {} {} features for all {} images in the DTD to {}'.format(thisRF, thisLayer, len(all_imgs), save_dir))
  return {'features': feature_maps, 'labels': labels}

###########################################################################################
########     
#######       MAIN FUNCTION
########
if __name__ == '__main__':
  tex_dir='/scratch/groups/jlg/tex_db_histmatch'
  save_dir = '/scratch/groups/jlg/texpca'
  layers = ['conv1_1', 'pool1', 'pool2', 'pool4']
  RFs = ['1x1', '2x2', '3x3', '4x4', '5x5', '6x6']

  layers = ['pool4']
  RFs = ['1x1']

  for ri, thisRF in enumerate(RFs):
    for li, thisLayer in enumerate(layers):
      # (1) Extract features (if you haven't already)
      if os.path.isfile('{}/{}_{}_DTD_features.npy'.format(save_dir, thisRF, thisLayer)):
        print('DTD features for {} {} already computed; Loading from savefile...'.format(thisRF, thisLayer))
        feats = np.load('{}/{}_{}_DTD_features.npy'.format(save_dir, thisRF, thisLayer))
        if feats.dtype=='O': # saved as dict instead of array
          feats = feats.item()['features']
      else:
        print('Running on RF: {} layer:{} '.format(thisRF, thisLayer))
        feats = get_feature_maps(thisLayer, thisRF, tex_dir=tex_dir, save_feature_maps=1, save_dir=save_dir)
        feats = feats['features']

      # (2) Compute PCA on those features
      if os.path.isfile('{}/{}_{}_PCs.npy'.format(save_dir, thisRF, thisLayer)):
        print('PCA already computed, skipping to next model')
        continue
      else:
        print('Done getting feature maps. Now running PCA...')
        pca = PCA(n_components=4000)
        print('Original features shape: {}'.format(feats.shape))
        pca.fit(feats.T)

        #pca_dict = {'pca': pca};

        print('PCA completed. Saving to {}/{}_{}_PCs.npy'.format(save_dir, thisRF, thisLayer))
        #np.save('{}/{}_{}_PCs.npy'.format(save_dir, thisRF, thisLayer), pca_dict)
