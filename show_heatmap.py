import pickle, time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob

from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque
from lesson_lib import *

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


# Get all the windows to be searched
all_search_windows = generate_search_windows(mpimg.imread('test_image.jpg'))

thresh = 3

history = []


def process_image(image):

    draw_image = np.copy(image)
    image = image.astype(np.float32) / 255.0

    hot_windows = search_windows(image, all_search_windows, svc, X_scaler, color_space=colorSpace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_depth, spatial_feat=useSpatial, 
                            hist_feat=useColor, hog_feat=True)


    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    box_list = hot_windows

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, thresh)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    return heatmap


for image in glob.glob('test_images/*'):

    f, axes = plt.subplots(2)
    axes[0].imshow(process_image(mpimg.imread(image)))
    axes[1].imshow(mpimg.imread(image))
    plt.show()
