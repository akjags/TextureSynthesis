from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

import copy, os, time
from pt_tex_synth import *
import argparse, pdb
from pt_synthesize import *

from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

###
tex_dir = '/scratch/groups/jlg/texture_db'

def get_feature_maps(thisLayer, tex_dir=tex_dir, save_feature_maps=1):
  # Get the pretrained VGG19 model
  cnn = models.vgg19(pretrained=True).features.to(device).eval()

  # This is the normalization mean and std for VGG19.
  cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
  cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

  # Specify which layers to match
  all_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4']
  this_layers = all_layers[:all_layers.index(thisLayer)+1]

  tStart = time.time()

  # Get the feature maps for each image in the tex_dir
  all_imgs = os.listdir(tex_dir)
  labels = []; fm = [];
  for i in range(len(all_imgs)):
    thisTex = all_imgs[i]
    img_path = tex_dir + '/' + thisTex
    style_img = image_loader(img_path)

    GMs = get_layer_features(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, style_layers=this_layers);

    # Extract gram matrix at this layer and unravel it into a vector.
    layer_features = GMs[thisLayer].cpu().numpy().ravel()

    # get the label and feature map
    fm.append(layer_features)
    labels.append(thisTex.split('_')[0])

    #pdb.set_trace()
    if i % 100 == 0:
      print(i, thisTex, '\t', layer_features.shape, '\t', labels[-1])

  tElapsed = time.time() - tStart
  print('Extracting {} feature maps for {} images took {} seconds'.format(thisLayer, len(all_imgs), tElapsed))
  feature_maps = np.stack(fm, axis=-1)

  if save_feature_maps:
    fmaps = {}
    fmaps['labels'] = labels
    fmaps['{}_features'.format(thisLayer)] = feature_maps
    np.save('/scratch/groups/jlg/texpca/{}_features_histmatch.npy'.format(thisLayer), fmaps)
  return feature_maps, labels


def run_pca():
  fm = np.load('/scratch/groups/jlg/texpca/pool2_features.npy').item()

  pca = PCA(n_components=10)
  pca.fit(fm['pool2_features'].T)
  explained_variance = pca.explained_variance_ratio_

  print('{} percent variance explained by first 10 features'.format(np.sum(explained_variance)))

tex_dir='/scratch/groups/jlg/tex_db_histmatch'
layers = ['conv1_1', 'pool1', 'pool2']

for li in range(len(layers)):
  thisLayer = layers[li]
  print('Running on layer: ', thisLayer)
  #fm, labels = get_feature_maps(thisLayer, tex_dir=tex_dir, save_feature_maps=1)
  fmaps = np.load('/scratch/groups/jlg/texpca/{}_features_histmatch.npy'.format(thisLayer)).item()
  fm,labels = fmaps['{}_features'.format(thisLayer)], fmaps['labels']

  print('Done getting feature maps. Now running PCA and LDA')
  X = np.abs(fm.T)
  nmf = NMF(n_components=20, init='random')
  nmf.fit(X)
  p2 = {};
  p2['nmf'] = nmf;

  #pca = PCA(n_components=20)
  #pca.fit(X)
  #lda = LinearDiscriminantAnalysis(n_components=20)
  #lda.fit(X,labels)

  #p2 = {};
  #p2['pca'] = pca
  #p2['lda'] = lda

  np.save('/scratch/groups/jlg/texpca/{}_dims_nmf.npy'.format(thisLayer), p2)
  print('Saving to /scratch/groups/jlg/texpca')
