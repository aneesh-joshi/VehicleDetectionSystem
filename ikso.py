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
from detection_lib import *

def extract_features(all_images, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial=32, histbin=5):
	print('EXTRACTING COLOR FEATURES')
	color_features = extract_color_features(all_images, spatial_size=(spatial, spatial), hist_bins=histbin)
	print('EXTRACTING HOG FEATURES')
	hog_features = extract_hog_features(all_images, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
	all_image_features = np.hstack((color_features, hog_features))
	return all_image_features

if __name__ == '__main__':
	vehicle_images = glob.glob('vehicle_data/vehicles/*/*.png', recursive=True)
	non_vehicle_images = glob.glob('vehicle_data/non-vehicles/*/*.png', recursive=True)

	print('FOUND :', len(vehicle_images), 'Vehicle Images')
	print('FOUND :', len(non_vehicle_images), 'Non Vehicle Images')

	sample_size = 3
	vehicle_images = vehicle_images[:sample_size]
	non_vehicle_images = non_vehicle_images[:sample_size]
	all_images = vehicle_images + non_vehicle_images

	# Labels
	y = np.hstack( (np.ones(len(vehicle_images)), np.zeros(len(non_vehicle_images))) )

	# PARAMETERS
	spatial = 32
	histbin = 5
	orient = 9
	cspace='RGB'
	pix_per_cell=8
	cell_per_block=2
	hog_channel='ALL'

	X = extract_features(all_images, hog_channel=hog_channel)
	print('Vehicle Features:', X.shape)

	k = extract_features(['test_image.jpg'], hog_channel=hog_channel)
	print('Vehicle Features:', k.shape)

	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	svc = LinearSVC()
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')

	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 10
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

	print('SAVING MODEL')
	with open('models/temp_model.pkl', 'wb') as f:
	    pickle.dump([svc, X_scaler], f)
	print('MODEL SAVED')