from skimage.io import imread
import os
import numpy as np
import sys

def preprocess_im(path):
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    image = imread(path)
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    if len(image.shape)<4:
      image = np.stack((image, image, image), axis=3)
    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

if __name__ == "__main__":
    # First, preprocess the images.
    args = sys.argv
    orig_im_dir = args[1]
    img_name = args[2]

    new_im = preprocess_im(orig_im_dir+'/'+img_name+'.png')
    np.save(orig_im_dir+'/'+img_name, new_im)
    print 'Preprocessed image ' + img_name + ' saved'
