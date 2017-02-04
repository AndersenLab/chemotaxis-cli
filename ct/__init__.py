import os
import numpy as np
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.draw import circle_perimeter
from skimage.morphology import binary_closing, binary_dilation, binary_erosion
from skimage.util import img_as_bool, img_as_int
from scipy import ndimage as ndi
from numpy import invert
from skimage.draw import circle
from skimage.io import imread
from clint.textui import puts_err, indent, colored
import matplotlib.pyplot as plt
import warnings
import hashlib
import pickle
from skimage.measure import regionprops
from matplotlib import colors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from sklearn import svm
from glob import glob


_program = "ct"
__version__ = "0.0.1"

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


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


def ci(v):
    """
        Calculate the chemotaxis index
    """
    return ((v[0] + v[3]) - (v[1] + v[2])) / float(v[6])

@memoize
def fit_model(X, y):
    """
        Fits model
    """
    clf = svm.SVC(verbose = False, kernel = 'linear')
    if X:
        clf.fit(X, y) 
        return clf
    else:
        return None

def load_model():
    """
        Loads training data and fits; caching for speed.
    """
    X_sets, y_sets = [], []
    for X in glob("train/X_*.data"):
        X_sets.extend(pickle.load(open(X, 'rb')))
    for y in glob("train/y_*.data"):
        y_sets.extend(pickle.load(open(y, 'rb')))
    
    return fit_model(X_sets, y_sets)


def save_training_set(X, y, fname):
    """
        Save individual traning set
    """
    make_dir("train")
    pickle.dump(X, open("train/X_" + fname + ".data", 'wb'))
    pickle.dump(y, open("train/y_" + fname + ".data", 'wb'))


@memoize
@suppress_warning
def find_plate(img, radii_range):
    """
        Identifies the location of the plate
    """
    # Read image, and convert to floating point
    img = img_as_bool(imread(img, mode="F", flatten=True))

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


class bbox:
    """
        Object for handling feature selection
    """


    def __init__(self, obj):
        self.t, self.l, self.b, self. r = obj.bbox
        self.label = obj.label

        # Construct properties
        properties = []
        properties.extend(obj.bbox) # Bounding box
        properties.append(obj.perimeter)
        properties.append(obj.area)
        properties.extend(obj.centroid)
        properties.append(obj.eccentricity)
        properties.append(obj.extent)
        properties.append(obj.filled_area)
        properties.extend(obj.inertia_tensor_eigvals)
        properties.append(obj.major_axis_length)
        properties.append(obj.orientation)
        properties.append(obj.solidity)
        self.properties = properties

    def in_box(self, x, y):
        return (self.t < y and y < self.b and self.l < x and x < self.r)

    def set_box(self, ax, mode = None):

        if mode == 'toggle':
            self.toggle = not self.toggle

        if self.toggle == True:
            self.color = '#0d99fc'
        else:
            self.color = '#ff2a1a'
        rect = mpatches.Rectangle((self.l, self.t), self.r - self.l, self.b - self.t,
                          fill=False, edgecolor=self.color, linewidth=2)
        ax.add_patch(rect)

    def __repr__(self):
        return "[{label}]".format(label = self.label)

@suppress_warning
def crop_and_filter_plate(img, radii_range, extra_crop, small = 100, large = 1200, debug= False, train = False):
    fname = os.path.splitext(os.path.basename(img))[0]
    center, radius = find_plate(img, radii_range)
    if debug:
        with indent(4):
            puts_err(colored.blue("Center: " + str(center[0]) + "," + str(center[1])))
            puts_err(colored.blue("Radius: " + str(radius)))
            puts_err(colored.blue("Cropping plate"))

    y, x = center

    # Filter background
    img = imread(img, flatten = True)
    t_crop = int(y - radius + extra_crop)
    b_crop = int(t_crop + radius*2 - extra_crop*2)
    l_crop = int(x - radius + extra_crop)
    r_crop = int(x + radius - extra_crop)
    img = img[t_crop:b_crop]
    img = img[:,l_crop:r_crop]

    img_out = img.copy()

    if debug:
        with indent(4):
            puts_err(colored.blue("Circular crop"))
        plt.imsave("debug/" + fname + ".05_crop.png", img)

    # Redefine x,y,radius; Generate circle mask.
    mask = np.zeros(img.shape, dtype=np.uint8)
    x, y, radius = [img.shape[0]/2] * 3

    rr, cc = circle(y, x, radius)
    mask[rr, cc] = 1
    img[mask == 0] = False

    if debug:
        with indent(4):
            puts_err(colored.blue("Performing edge detection"))
        plt.imsave("debug/" + fname + ".06_mask.png", img)

    # Apply a canny filter
    img = canny(img, sigma=1.5, mask = mask == 1, low_threshold = 0.20, high_threshold = 0.30)

    # Remove the edge
    mask = np.zeros(img.shape, dtype=np.uint8)
    rr, cc = circle(y, x, radius-3)
    mask[rr, cc] = 1
    img[mask == 0] = False

    if debug:
        with indent(4):
            puts_err(colored.blue("Binary  Dilation"))
        plt.imsave("debug/" + fname + ".07_edges.png", img, cmap='copper')

    # Dilate
    img = binary_dilation(binary_closing(img))

    if debug:
        with indent(4):
            puts_err(colored.blue("Binary Fill"))
        plt.imsave("debug/" + fname + ".08_dilation.png", img, cmap='copper')

    # Fill edges
    img = ndi.binary_fill_holes(img)

    if debug:
        with indent(4):
            puts_err(colored.blue("Apply filters"))
        plt.imsave("debug/" + fname + ".09_fill.png", img, cmap='copper')

    # Remove small particles
    label_objects, nb_labels = ndi.label(img)
    sizes = np.bincount(label_objects.ravel())

    # Label by mask
    reg_props = regionprops(label_objects)
    axis_length = np.array([x.minor_axis_length for x in reg_props])

    # Apply SVM
    model = load_model()
    if model is None:
        with indent(4):
            puts_err(colored.red("You need to train first!"))

    # Apply SVM
    objects = []
    for reg in reg_props:
        box = bbox(reg)
        pred = model.predict(box.properties)[0]
        box.toggle = pred
        objects.append(box)

    if train:     
        fig = plt.gcf()
        ax = plt.gca()

        # remove artifacts connected to image border
        cleared = img_out.copy()
        clear_border(cleared)

        # label image regions
        label_image = label(cleared)
        borders = np.logical_xor(img_out, cleared)
        label_image[borders] = -1        
        image_label_overlay = label2rgb(label_image, image=img_out)
        ax.imshow(image_label_overlay)

        for box in objects:
            box.set_box(ax)

        im = plt.imshow(img_out, cmap = 'Greys')

        class EventHandler:
            def __init__(self):
                fig.canvas.mpl_connect('button_press_event', self.onpress)

            def onpress(self, event):
                if event.inaxes!=ax:
                    return
                xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
                value = im.get_array()[xi,yi]
                color = im.cmap(im.norm(value))
                [x.set_box(ax, 'toggle') for x in objects if x.in_box(xi, yi)]
                fig.canvas.draw()

        handler = EventHandler()

        with indent(4):
            puts_err(colored.blue("\nClick non-worm objects (set bounding box to red). Exit when done.\n"))

        plt.show()

        # Save training data
        X = [x.properties for x in objects]
        y = [x.toggle for x in objects]
        save_training_set(X, y, fname)

    # apply filters
    svm_filter = np.array([True] + [x.toggle for x in objects])

    # Label filters
    filters = np.zeros(len(reg_props)+1, dtype='int32')
    filters[filters == 0] = 2 # KEEP
    filters[svm_filter == False] = 1 # FILTER
    filters[0] = 0 # Background

    filter_img = label_objects.copy()
    for k,v in enumerate(filters):
        filter_img[filter_img == k] = v

    if debug:
        cmap = colors.ListedColormap(['white', 'red', 'gray'])
        bounds=[0,1,2]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imsave("debug/" + fname + ".10_filters.png", filter_img, cmap=cmap)

    # Apply filter
    filters = filters == 2
    filters[0] = 0
    img = filters[label_objects]

    if debug:
        plt.imsave("debug/" + fname + ".11_filtered.png", img, cmap='copper')

    return img


def pixel_counts(img, n_radius_divisor):
    r = img.shape[0]/2

    mask = np.zeros(img.shape, dtype=np.uint8)
    x, y, radius = [img.shape[0]/2] * 3
    radius = radius / n_radius_divisor
    rr, cc = circle(y, x, radius)
    mask[rr, cc] = 1

    n, q = img.copy(), img.copy()
    q[mask == 1] = False
    n[mask == 0] = False

    tl = sum(q[0:r,0:r].flatten())
    tr = sum(q[0:r,r:].flatten())
    bl = sum(q[r:,0:r].flatten())
    br = sum(q[r:,r:].flatten())
    n = sum(n.flatten())
    total_q = sum(q.flatten())
    total = sum(img.flatten())
    ret = [tl, tr, bl, br, n, total_q, total]
    ci_val = ci(ret)
    return ret + [ci_val]


