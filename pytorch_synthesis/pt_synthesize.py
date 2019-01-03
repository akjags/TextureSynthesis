from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy, os, time
from pt_tex_synth import *
import argparse
import pdb

# desired size of the output image
imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    
def imsave(tensor, savepath=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(savepath)    

########## MAIN: SPECIFY OPTIONS:
if __name__ == "__main__":
  ### Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--img_path", default="/home/users/akshayj/TextureSynthesis/stimuli/textures/orig_color/cherries.jpg", help="specifies the path of the original image")
  parser.add_argument("-o", "--out_dir", default="/home/users/akshayj/TextureSynthesis/stimuli/textures/out_color/v2", help="specifies the path of the output directory")
  parser.add_argument("-l", "--layer", default="pool4", help="specifies the layer to match statistics through")
  parser.add_argument("-s", "--nSplits", default=1, help="specifies the number of sections to split each dimension into (e.g. 1x1, 2x2, nxn)")
  parser.add_argument('-n', '--nSteps', type=int, default=10000, help="specifies the number of steps to run the gradient descent for")
  args = parser.parse_args()

  ### PyTorch Models (VGG)
  # Load the model (VGG19)
  cnn = models.vgg19(pretrained=True).features.to(device).eval()

  # This is the normalization mean and std for VGG19.
  cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
  cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

  ### Specify texture synthesis parameters.
  # Load the original image
  style_img = image_loader(args.img_path)
  img_name = args.img_path.split('/')[-1].split('.')[0] # get just the name (e.g. cherries) without path or extension

  # Specify which layers to match
  all_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'];
  this_layers = all_layers[:all_layers.index(args.layer)+1]
  #this_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'];
  print(this_layers)

  ## Run Texture Synthesis
  # Randomly initialize white noise input image
  input_img = torch.randn(style_img.data.size(), device=device)

  # Make directory if it doesn't already exist
  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir) 
  saveName = '{}x{}_{}_{}.png'.format(args.nSplits, args.nSplits, args.layer, img_name) # e.g. 1x1_pool2_cherries.png

  # Get layer features
  get_layer_features(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, style_layers=this_layers);

  # Check if we're on GPU or CPU, then run! 
  gpu_str = "Using GPU" if torch.cuda.is_available() else "Using CPU"
  print("{} to synthesize textures at layer {}, nSplits: {}, image: {}, numSteps: {}".format(gpu_str, args.layer, args.nSplits, img_name, args.nSteps))
  tStart = time.time()
  output_leaves = run_texture_synthesis(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, input_img, num_steps=args.nSteps, style_layers=this_layers, saveLoc=[args.out_dir, saveName])

  tElapsed = time.time() - tStart
  print('Done! {} steps took {} seconds. Saving as {} now.'.format(args.nSteps, tElapsed, saveName))
  # Save final product to output directory
  imsave(output_leaves, args.out_dir + '/' + saveName);

