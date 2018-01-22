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
colorSpace = 'YUV'#params['colorSpace']
spatial_size = params['spatial_size']
hist_bins = params['hist_bins']
        


for image in glob.glob('test_images/*.jpg'):

    image = mpimg.imread(image)

    if isJpeg:
        image = image.astype(np.float32) / 255.0

    x_start_stop = [None, None]
    y_start_stop = [390, None]
    xy_overlap = [0.6, 0.5]
    xy_window = [128, 128]

    windows = slide_window(image,
                            x_start_stop=x_start_stop, 
                            y_start_stop=y_start_stop, 
                            xy_window=xy_window, 
                            xy_overlap=xy_overlap)

    x_start_stop = [None, None]
    y_start_stop = [390, None]
    xy_overlap = [0.6, 0.5]
    xy_window = [96, 96]

    windows += slide_window(image,
                            x_start_stop=x_start_stop, 
                            y_start_stop=y_start_stop, 
                            xy_window=xy_window, 
                            xy_overlap=xy_overlap)

    x_start_stop = [None, None]
    y_start_stop = [390, None]
    xy_overlap = [0.6, 0.5]
    xy_window = [64, 64]


    windows += slide_window(image,
                            x_start_stop=x_start_stop, 
                            y_start_stop=y_start_stop, 
                            xy_window=xy_window, 
                            xy_overlap=xy_overlap)

    x_start_stop = [None, None]
    y_start_stop = [390, None]
    xy_overlap = [0.6, 0.5]
    xy_window = [96, 96]

    windows += slide_window(image,
                            x_start_stop=x_start_stop, 
                            y_start_stop=y_start_stop, 
                            xy_window=xy_window, 
                            xy_overlap=xy_overlap)

    hot_windows = search_windows(image, windows, svc, X_scaler, 
                                color_space=colorSpace, 
                                spatial_size=spatial_size,
                                hist_bins=hist_bins, 
                                orient=orient, 
                                pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_depth, 
                                spatial_feat=useSpatial, 
                                hist_feat=useColor,
                                hog_feat=True)

    draw_image = draw_boxes(image, hot_windows)

    plt.imshow(draw_image)
    plt.show()
