import os
import numpy as np
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.draw import circle_perimeter
from skimage.morphology import binary_dilation
from skimage.util import img_as_bool
from scipy import ndimage as ndi
from numpy import invert
from skimage.draw import circle
from skimage.io import imread
from clint.textui import puts, indent, colored
import matplotlib.pyplot as plt
import warnings
import hashlib


import os, pickle
def memoize(func):
    def decorated(*args, **kwargs):
        if not os.path.exists('_cache'):
            os.makedirs('_cache')
        digest = hashlib.md5(pickle.dumps([args, kwargs])).hexdigest()
        cache_fname = '_cache/' + digest + ".pkl"
        if os.path.exists(cache_fname):
            with open(cache_fname) as f:
                cache = pickle.load(f)
        else:
            cache = func(*args)
            # update the cache file
            with open(cache_fname, 'wb') as f:
                pickle.dump(cache, f)
        return cache
    return decorated


def suppress_warning(f):
    def decorated(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f(*args, **kwargs)
    return decorated

canny = memoize(canny)

_program = "ct"
__version__ = "0.0.1"

@memoize
@suppress_warning
def find_plate(img, radii_range):
    """
        Identifies the location of the plate
    """
    # Read image, and convert to floating point
    img = invert(img_as_bool(imread(img, mode="F", flatten=True)))

    # Detect edges
    edges = canny(img, sigma=2)

    # Find circles
    hough_radii = np.arange(radii_range[0], radii_range[1], 2)
    hough_res = hough_circle(edges, hough_radii)

    centers, accums, radii = [], [], []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        num_peaks = 1
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    center = np.mean(centers, axis=0)
    radius = (sum(radii) * 1.0) / len(radii)
    return center, radius

@suppress_warning
def crop_and_filter_plate(img, radii_range, extra_crop, small_obj_size = 100, debug= False):
    fname = os.path.splitext(os.path.basename(img))[0]
    center, radius = find_plate(img, radii_range)
    if debug:
        with indent(4):
            puts(colored.blue("Center: " + str(center[0]) + "," + str(center[1])))
            puts(colored.blue("Radius: " + str(radius)))
            puts(colored.blue("Cropping plate"))

    y, x = center

    # Filter background
    img = imread(img, flatten = True)
    t_crop = y - radius + extra_crop
    b_crop = t_crop + radius*2 - extra_crop*2
    l_crop = x - radius + extra_crop
    r_crop = x + radius - extra_crop
    img = img[t_crop:b_crop]
    img = img[:,l_crop:r_crop]

    if debug:
        with indent(4):
            puts(colored.blue("Circular crop"))
        plt.imsave("debug/" + fname + ".05_crop.png", img)

    # Redefine x,y,radius; Generate circle mask.
    mask = np.zeros(img.shape, dtype=np.uint8)
    x, y, radius = [img.shape[0]/2] * 3

    rr, cc = circle(y, x, radius)
    mask[rr, cc] = 1
    img[mask == 0] = False

    if debug:
        with indent(4):
            puts(colored.blue("Performing edge detection"))
        plt.imsave("debug/" + fname + ".06_mask.png", img)

    # Apply a canny filter
    img = canny(img, sigma=1.5, mask = mask == 1, low_threshold = 0.05, high_threshold = 0.20)

    # Remove the edge
    mask = np.zeros(img.shape, dtype=np.uint8)
    rr, cc = circle(y, x, radius-3)
    mask[rr, cc] = 1
    img[mask == 0] = False

    if debug:
        with indent(4):
            puts(colored.blue("Binary  Dilation"))
        plt.imsave("debug/" + fname + ".07_edges.png", img, cmap='copper')

    # Dilate
    img = binary_dilation(img)

    if debug:
        with indent(4):
            puts(colored.blue("Binary Fill"))
        plt.imsave("debug/" + fname + ".08_dilation.png", img, cmap='copper')

    # Fill edges
    img = ndi.binary_fill_holes(img)

    if debug:
        with indent(4):
            puts(colored.blue("Filter small particles"))
        plt.imsave("debug/" + fname + ".09_fill.png", img, cmap='copper')

    # Remove small particles
    label_objects, nb_labels = ndi.label(img)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > small_obj_size
    mask_sizes[0] = 0
    img = mask_sizes[label_objects]

    if debug:
        plt.imsave("debug/" + fname + ".10_filter_small.png", img, cmap='copper')

    return img

