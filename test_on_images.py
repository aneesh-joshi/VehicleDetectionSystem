import pickle, glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from lesson_lib import *

# PARMAS====================================================
isJpeg = True

# ====================================================

with open('Models/model_YUV0_hog_hist_spa.pkl', 'rb') as f:
    svc, X_scaler, params = pickle.load(f)

orient = params['orient']
pix_per_cell = params['pix_per_cell']
cell_per_block = params['cell_per_block']
hog_depth = params['hog_depth']
useColor = params['useColor']
useSpatial = params['useSpatial']
colorSpace = params['colorSpace']
spatial_size = params['spatial_size']
hist_bins = params['hist_bins']


scale = 2
ystart = 300
ytop = 700

for test_image in glob.glob('test_images/*'):
    image = mpimg.imread(test_image)
    draw_image = np.copy(image)

    if (isJpeg):
        image = image.astype(np.float32)/255.0

    ctrans_tosearch = convert_color(image, conv=colorSpace)

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64  
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    if hog_depth == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog_features = np.hstack((hog1.ravel(), hog2.ravel(), hog3.ravel()))
    else:
        hog_features = get_hog_features(ctrans_tosearch[:,:,hog_depth], orient, pix_per_cell, cell_per_block, feature_vec=False)

    rectangles = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            if hog_depth == 'ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).reshape(1,-1)
            else:
                # TODO make dynamic
                features = hog_features[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # print(features.shape)
            if useSpatial:
                spatial_features = bin_spatial(subimg, size=spatial_size).reshape(-1,)
                # print(spatial_features.shape)
                features = np.hstack((features, spatial_features)).reshape((1, -1))

            if useColor:
                hist_features = color_hist(subimg, nbins=hist_bins).reshape((1, -1))
                features = np.hstack((features, hist_features)).reshape((1, -1))

            # Scale features and make a prediction
            # test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))    
            test_features = features  
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
        
    draw_image = draw_boxes(draw_image, rectangles)
    
    plt.imshow(draw_image)
    plt.show()