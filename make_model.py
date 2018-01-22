import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import pickle

from lesson_lib import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__ == '__main__':
    data_path = 'vehicle_data/'

    cars = glob.glob(data_path + 'vehicles/*/*.png')
    notcars = glob.glob(data_path + 'non-vehicles/*/*.png')

    print('FOUND %d Vehicle images' % len(cars))
    print('FOUND %d Non Vehicle images' % len(notcars))

    sample_size = -1
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]
    all_images = cars + notcars

    print('Training on a sample size of %d' % sample_size)

    y = np.hstack([np.ones(len(cars)), np.zeros(len(notcars))])

    # PARAMETERS
    #============================================================
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    hog_depth = 0

    useColor = True
    useSpatial = True

    spatial_size = (32, 32)
    hist_bins = 5

    colorSpace = 'RGB2YUV'
    #============================================================

    params = dict([('orient', orient), 
        ('pix_per_cell', pix_per_cell), 
        ('cell_per_block', cell_per_block),
        ('useColor', useColor),
        ('useSpatial', useSpatial),
        ('colorSpace', colorSpace),
        ('hog_depth', hog_depth),
        ('spatial_size', spatial_size),
        ('hist_bins', hist_bins)])

    X = []

    for i, image in enumerate(all_images):
        image = mpimg.imread(image)

        ctrans_tosearch = convert_color(image, conv=colorSpace)

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
        
        if hog_depth == 'ALL':
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
            features = np.hstack((hog1.ravel(), hog2.ravel(), hog3.ravel()))
        else:
            features = get_hog_features(ctrans_tosearch[:,:,hog_depth], orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()

        if useSpatial:
            spatial_features = bin_spatial(ctrans_tosearch, size=spatial_size).reshape(-1,)
            features = np.hstack((features, spatial_features)).reshape((1, -1))

        if useColor:
            hist_features = color_hist(ctrans_tosearch, nbins=hist_bins).reshape((1, -1))
            features = np.hstack((features, hist_features)).reshape((1, -1))

        features = features.reshape(-1)
        X.append(features)
        
    X = np.array(X)


    print('X, y shapes:')
    print(X.shape, y.shape)

    X, y = shuffle(X, y)
    print('SHUFFLE COMPLETE')

    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)


    svc = LinearSVC()
    svc = svc.fit(X_train, y_train)

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    with open('Models/model_col_hog.pkl', 'wb') as f:
        pickle.dump([svc, X_scaler, params], f)

    print('MODEL SAVED')
