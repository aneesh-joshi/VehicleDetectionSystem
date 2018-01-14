import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from detection_lib import *
from ikso import extract_features



with open('models/temp_model.pkl', 'rb') as f:
    svc, X_scaler = pickle.load(f)
print('PICKLE LOADED')

image = ['test_image.jpg']


# PARAMETERS
spatial = 32
histbin = 5
orient = 9
cspace='RGB'
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'

X = extract_features(image, hog_channel=hog_channel, spatial=spatial, histbin=histbin)
print('Image Features:', X.shape)