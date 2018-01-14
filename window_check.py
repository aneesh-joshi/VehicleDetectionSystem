import glob
import pickle
from new_svm_configure import *
from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


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

xy_overlap=(0.5,0.9)
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

	# plt.imshow(window_img)
	# plt.show()

	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	box_list = hot_windows

	# Add heat to each box in box list
	heat = add_heat(heat,box_list)
	    
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat, 2)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(image), labels)

	plt.imshow(draw_img)
	plt.show()

	plt.imshow(heatmap)
	plt.show()
	

lel
# Read in a pickle file with bboxes saved
# Each item in the "all_bboxes" list will contain a 
# list of boxes for one of the images shown above
box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

# Read in image similar to one shown above 
image = mpimg.imread('test_image.jpg')
heat = np.zeros_like(image[:,:,0]).astype(np.float)

# Add heat to each box in box list
heat = add_heat(heat,box_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()