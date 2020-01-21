from skimage.io import imread
from skimage.transform import resize
import os, time
import numpy as np

def preprocess_im(path):
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    image = imread(path)

    if image.shape[1]!=256 or image.shape[0]!=256:
        image = resize(image, (256,256))

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    if len(image.shape)<4:
        image = np.stack((image,image,image),axis=3)

    # If there is a Alpha channel, just scrap it
    if image.shape[3] == 4:
        image = image[:,:,:,:3]

    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

if __name__ == "__main__":
    # First, preprocess the images.
    orig_im_dir = '/scratch/groups/jlg/seibert/hvm_images'
    out_dir = orig_im_dir

    orig_ims = os.listdir(orig_im_dir)

    print('Preprocessing PNG and JPG images in {}'.format(orig_im_dir))
    start_time = time.time()
    for i, im in enumerate(orig_ims):
        if '.png' in im or '.jpg' in im:
            imName = '.'.join(im.split('.')[:-1]) # get rid of the extension
            if imName not in orig_ims:
                new_im = preprocess_im(orig_im_dir+'/'+im)
                np.save(out_dir+'/'+imName, new_im)
        if i % 100 ==0:
            print('{}% complete'.format(100.0*i/len(orig_ims)))
    elapsed_time = time.time() - start_time
    print('Preprocessing complete! Took {} seconds'.format(elapsed_time))
