import glob
import pickle
from new_svm_configure import *

with open('model.pkl', 'rb') as f:
	svc, X_scaler = pickle.load(f)


# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)


color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block

hog_channel = 0 # Can be 0, 1, 2, or "ALL"

spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 5    # Number of histogram bins
spatial_feat = True # Spatial features on or off

hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [360, 720] # Min and max in y to search in slide_window()

xy_overlap=(0.5,0.8)
xy_window =(96, 96)

for image_path in glob.glob('test_images/*'):
	image = mpimg.imread(image_path)
	draw_image = np.copy(image)
	image = image.astype(np.float32)/255

	windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
	                    xy_window=xy_window, xy_overlap=xy_overlap)

	hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
	                        spatial_size=spatial_size, hist_bins=hist_bins, 
	                        orient=orient, pix_per_cell=pix_per_cell, 
	                        cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
	                        hist_feat=hist_feat, hog_feat=hog_feat)                       

	window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

	plt.imshow(window_img)
	plt.show()
