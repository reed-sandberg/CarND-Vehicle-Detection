#!/usr/bin/env python

# Train an SVM classifier from a set of images labeled as either a car image or not a car image.
# Pickle the results and dump to svc_pickle.p

import glob
import numpy as np
import pickle
import time

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from lesson_functions import *


def extract_features(img_file_names, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """Extract features from a list of images and return a list of feature vectors - one vector per image.
    Each feature vector is a combination of features for each desired feature type in the order (spatial color,
    color histogram, HOG) according to the extraction params given."""

    # List of feature vectors (one for each input image).
    features = []

    for img_file_name in img_file_names:
        image = mpimg.imread(img_file_name)

        # Apply a color model conversion if other than 'RGB'.
        feature_image = as_color_model(image, color_space)

        # Vector of features for this image.
        img_features = []

        # Append features of each type in the specific order.
        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            img_features.append(hist_features)
        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block,
                                                vis=False, feature_vec=True)
            img_features.append(hog_features)
        features.append(np.concatenate(img_features))
    return features


# Read the training images.
notcar_fnames = glob.glob('non-vehicles_smallset/notcars*/*.jpeg') 
car_fnames = glob.glob('vehicles_smallset/cars*/*.jpeg')

# Don't modify these without considering the prediction routines that use the classifier.
# Color channel for HOG analysis.
hog_channel = 'ALL'
# Spatial features on or off.
spatial_feat = True
# Histogram features on or off.
hist_feat = True
# HOG features on or off.
hog_feat = True

car_features = extract_features(car_fnames, color_space=HOG_CSPACE, 
                                spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                                orient=HOG_ORIENTATIONS, pix_per_cell=PIX_PER_CELL, 
                                cell_per_block=CELL_PER_BLOCK, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcar_fnames, color_space=HOG_CSPACE, 
                                   spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                                   orient=HOG_ORIENTATIONS, pix_per_cell=PIX_PER_CELL, 
                                   cell_per_block=CELL_PER_BLOCK, 
                                   hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                   hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors for the entire training set.
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Create the labels vector.
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets.
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler for feature vector normalization across feature types.
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to normalize feature vectors.
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:', HOG_ORIENTATIONS, 'orientations', PIX_PER_CELL, 'pixels per cell and', CELL_PER_BLOCK,
      'cells per block')
print('Feature vector length:', len(X_train[0]))

# Create the classifier using a linear SVC.
svc = LinearSVC(C=0.01)

# Train the classifier with timing.
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC.
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Store the pickled classifier and normalization scaler.
dist_pickle = {'svc': svc, 'scaler': X_scaler}
pickle.dump(dist_pickle, open('./svc_trained.p', 'wb'))
