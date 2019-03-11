import numpy as np
import pdb, os
from skimage.io import imread, imsave

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def normalize_images(tex_dir, out_dir):
    """
    Equalize histograms of images.
    """
    images = os.listdir(tex_dir)

    targetR = np.zeros((0,0)) # initialize empty array
    targetG = np.zeros((0,0))
    targetB = np.zeros((0,0))
    print('Collecting image histograms')
    for imName in images[:100]:
        img = imread('{}/{}'.format(tex_dir, imName))
        if len(img.shape)==1:
            img = np.stack((img,img,img),2)
        targetR = np.append(targetR, img[:,:,0].ravel())
        targetG = np.append(targetG, img[:,:,1].ravel())
        targetB = np.append(targetB, img[:,:,2].ravel())

    print('Done collecting image histograms. Now, matching histograms')
    for imName in images:
        if imName in os.listdir(out_dir):
            print('{} already computed. Skipping..'.format(imName))
            continue
        img = imread('{}/{}'.format(tex_dir, imName))
        if len(img.shape)==1:
            img = np.stack((img,img,img),2)
        new_img = np.zeros((img.shape[0], img.shape[1], 3))
        new_img[:,:,0] = hist_match(img[:,:,0], targetR)
        new_img[:,:,1] = hist_match(img[:,:,1], targetG)
        new_img[:,:,2] = hist_match(img[:,:,2], targetB)
        imsave('{}/{}'.format(out_dir, imName), new_img.astype(int))
        print('Saved {} to {}'.format(imName, out_dir))
    print('Done matching histograms and saved to output directory')


tex_dir = '/scratch/groups/jlg/texture_db'
out_dir = '/scratch/groups/jlg/tex_db_histmatch'
normalize_images(tex_dir, out_dir)

