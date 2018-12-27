from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
from pt_tex_synth import *
import argparse

# desired size of the output image
imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image


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
  ## Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--path", help="specify the path of the original image")
  parser.add_argument("-o", "--outdir", help="specify the path of the output directory")
  parser.add_argument("-l", "--layer", help="specify the layer to match statistics through")
  parser.add_argument("-n", "--nSplits", help="specify the number of splits")
  args = parser.parse_args()

  # Load the model (VGG19)
  cnn = models.vgg19(pretrained=True).features.to(device).eval()

  # This is the normalization mean and std for VGG19.
  cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
  cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

  # Specify num_steps
  num_steps = 5000

  # Specify the style image to match
  style_img = image_loader("/home/users/akshayj/TextureSynthesis/stimuli/textures/orig_color/cherries.jpg")

  # Specify which layers
  style_layers = ['conv_1_1', 'pool_1', 'pool_2', 'pool_4'];

  # Randomly initialize white noise input image
  input_img = torch.randn(style_img.data.size(), device=device)

  output_leaves = run_texture_synthesis(cnn, cnn_normalization_mean, cnn_normalization_std,
                              style_img, input_img, num_steps=num_steps, style_layers=style_layers)

  imsave(output_leaves, 'out_cherries2.png');

