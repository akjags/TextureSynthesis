from skimage.io import imread, imsave
import os
import numpy as np


def postprocess_im(img):
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    image = img + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    return image

if __name__ == "__main__":
    # POST PROCESSING: Resave all npy images as jpg's
    outDirs = ['rf_stim/s1', 'rf_stim/s2', 'rf_stim/s3']

    for outDir in outDirs:
        print outDir
        for im in os.listdir(outDir):
            if '.npy' in im:
                imName = outDir+'/'+im
                #imi = postprocess_im(np.load(imName))
                imi = np.load(imName)
                imsave(outDir+'/'+im[:-4]+'.jpg', imi)
                print im + ' saved as jpg' 
