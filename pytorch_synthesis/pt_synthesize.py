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

# desired size of the output image
imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu
print(device)

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001) # pause a bit so that plots are updated

def imsave(tensor, savepath=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(savepath)    

# Load the model 
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Specify the normalization mean and std.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Randomly initialize white noise input image
input_img = torch.randn(style_img.data.size(), device=device)
orig_input = input_img.cpu().clone(); # save it to a different filename

########## MAIN: SPECIFY OPTIONS:

# Specify num_steps
num_steps = 1000

# Specify the style image to match
style_img = image_loader("/Users/akshay/proj/TextureSynthesis/stimuli/textures/orig_color/cherries.jpg")

# Specify which layers
style_layers = ['conv_1', 'conv_2'];
style_layers = ['pool1', 'pool2', 'pool4'];

output_leaves = run_texture_synthesis(cnn, cnn_normalization_mean, cnn_normalization_std,
                            style_img, input_img, num_steps=num_steps, style_layers=style_layers_default)

imsave(output_leaves, 'out_cherries2.png');

