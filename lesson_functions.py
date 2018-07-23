#!/usr/bin/env python

import numpy as np
import cv2

import matplotlib.pyplot as plt

from skimage.feature import hog


# HOG color model.
HOG_CSPACE = 'YCrCb'

# Number of HOG orientation buckets.
HOG_ORIENTATIONS = 9

# HOG pixels per cell.
PIX_PER_CELL = 8

# HOG cells per block.
CELL_PER_BLOCK = 2

# Spatial binning dimensions.
SPATIAL_SIZE = (32, 32)

# Number of histogram bins for color histogram features.
HIST_BINS = 32


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """Return HOG features and an optional image for feature visualization as a 2-tuple if vis is True."""
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               block_norm= 'L2-Hys',
               transform_sqrt=True,
               visualise=vis, feature_vector=feature_vec)

def bin_spatial(img, size=(32, 32)):
    """Aggregate channel color values into the given size dimensions using a simple resize() function to bin color
    channel values, and ravel() to create the feature vector as 1-D."""
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0, 255)):
    """Determine a histogram of color values for each color channel and bundle them into a 1-D feature vector."""

    # Compute the histogram of each color channel separately.
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector.
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

def as_color_model(img, color_space):
    """Copy and convert img to new a color model (color_space). This function assumes img is represented
    in 'RGB', so if color_space='RGB', simply return a copy of img as-is."""
    if color_space == 'RGB':
        feature_image = np.copy(img)
    else:
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return feature_image

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw boundaries of each set of rectangular coordinates from bboxes onto a copy of img."""
    # Make a copy of the image.
    imcopy = np.copy(img)
    # Iterate through the bounding boxes.
    for bbox in bboxes:
        # Draw a rectangle given the bbox coordinates.
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with the boxes drawn.
    return imcopy

def plot_image_bundle(fig, row_cnt, col_cnt, imgs, titles, cmap='hot'):
    """Plot imgs on a single figure (canvas) of row_cnt x col_cnt with titles displayed."""
    for img_idx, img in enumerate(imgs):
        plt.subplot(row_cnt, col_cnt, img_idx + 1)
        plt.title(img_idx + 1)
        if len(img.shape) < 3:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        plt.title(titles[img_idx])
