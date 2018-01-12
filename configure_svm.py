import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import sys
from pprint import pprint
import pickle

import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from detection_lib import color_hist, bin_spatial, get_hog_features, extract_color_features, extract_hog_features

vehicle_images = glob.glob('vehicle_data/vehicles/*/*.png', recursive=True)
non_vehicle_images = glob.glob('vehicle_data/non-vehicles/*/*.png', recursive=True)
# print(vehicle_images)
print('FOUND :', len(vehicle_images), 'Vehicle Images')
print('FOUND :', len(non_vehicle_images), 'Non Vehicle Images')


def process_image(image, params):



spatial = 32
histbin = 5

params['spatial'] = spatial
params['histbin'] = histbin

vehicle_color_features = extract_color_features(vehicle_images, spatial_size=(spatial, spatial), hist_bins=histbin)
non_vehicle_color_features = extract_color_features(non_vehicle_images, spatial_size=(spatial, spatial), hist_bins=histbin)

vehicle_features = np.hstack((extract_hog_features(vehicle_images), vehicle_color_features))
non_vehicle_features = np.hstack((extract_hog_features(non_vehicle_images), non_vehicle_color_features))

X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# y = np.hstack( (np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))) )


# # Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

# print('Using spatial binning of:',spatial, 'and', histbin,'histogram bins')
# print('Feature vector length:', len(X_train[0]))

# # Use a linear SVC
# svc = LinearSVC()
# # Check the training time for the SVC
# t=time.time()
# svc.fit(X_train, y_train)
# t2 = time.time()
# print(round(t2-t, 2), 'Seconds to train SVC...')
# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
# t=time.time()
# n_predict = 10
# print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
# print('For these',n_predict, 'labels: ', y_test[0:n_predict])
# t2 = time.time()
# print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# print('SAVING MODEL')
# with open('models/temp_model.pkl', 'wb') as f:
#     pickle.dump([svc, X_scaler], f)
# print('MODEL SAVED')
