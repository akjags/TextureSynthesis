import numpy as np
from pt_tex_synth import *
from pt_synthesize import *
import os

def get_save_features(retrieve, stim_dir, save_dir):

  if retrieve=='activations':
    feature_type = 'activation'
    feature_func = get_layer_activations
  else:
    feature_type = 'gram'
    feature_func = get_layer_features

  print('Getting {} features from images in {} and saving to {}'.format(feature_type, stim_dir, save_dir))

  layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4']
  cnn = models.vgg19(pretrained=True).features.to(device).eval()
  normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
  normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

  imgs = os.listdir(stim_dir)
  for i, im in enumerate(imgs):
    savepath = '{}/{}_{}.npy'.format(save_dir, feature_type, im.split('.')[0])
    if os.path.isfile(savepath):
      print('{} already found, skipping...'.format(savepath))
      continue
    print('{} of {}: Computing and saving gram matrix for image {}'.format(i+1, len(imgs), im))
    style_img = image_loader(stim_dir + '/' + im)
    sl = feature_func(cnn, normalization_mean, normalization_std, style_img, style_layers=layers)
    np.save(savepath, sl);

if __name__=='__main__':
  stim_dir = '/scratch/users/akshayj/tex_fMRI/tex_eq'
  save_dir = '/scratch/users/akshayj/tex_fMRI/gram_eq'
  feature_type = 'gramian'

  get_save_features(feature_type, stim_dir, save_dir)
