import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image = mpimg.imread('test_image.jpg')
YUV = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
LUV = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

f, axes = plt.subplots(3,3)

axes[0,0].imshow(YUV[:,:,0], cmap='gray')
axes[0,0].set_title('Y')
axes[0,1].imshow(YUV[:,:,1], cmap='gray')
axes[0,1].set_title('U')
axes[0,2].imshow(YUV[:,:,2], cmap='gray')
axes[0,2].set_title('V')

axes[1,0].imshow(YCrCb[:,:,0], cmap='gray')
axes[1,0].set_title('Y')
axes[1,1].imshow(YCrCb[:,:,1], cmap='gray')
axes[1,1].set_title('Cr')
axes[1,2].imshow(YCrCb[:,:,2], cmap='gray')
axes[1,2].set_title('Cb')

axes[2,0].imshow(LUV[:,:,0], cmap='gray')
axes[2,0].set_title('L')
axes[2,1].imshow(LUV[:,:,1], cmap='gray')
axes[2,1].set_title('U')
axes[2,2].imshow(LUV[:,:,2], cmap='gray')
axes[2,2].set_title('V')

# plt.imshow(YUV)
plt.show()