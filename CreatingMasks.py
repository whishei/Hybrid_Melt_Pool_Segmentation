#UTEP Project 

#Creating Training Data

import numpy as np
import skimage
from PIL import Image
from skimage.util import invert
from skimage import data
from skimage.filters import meijering, hessian,gaussian
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

#Opening the raw image
image_raw = Image.open('Testing/crop45.tif')
image = image_raw.convert('L')
image = np.asarray(image)

#Visualization
plt.imshow(image, cmap = plt.cm.gray)
plt.show()

#Passing through a Gaussian Filter
image = ndi.gaussian_filter(image, 2)

#Visualization
plt.imshow(image, cmap = plt.cm.gray)
plt.show()

#Passing through Meijering or Hessian Filter
result = meijering(image, black_ridges = 0, sigmas = [1]) #hessian(image,black_ridges=0,sigmas=1) 

#Visualization
plt.imshow(result, cmap = plt.cm.gray)
#plt.axis('off')
plt.show()

#Filling Holes
thresh_value = skimage.filters.threshold_otsu(result)
thresh = result < 0.15 #thresh_value
fill = ndi.binary_fill_holes(thresh)

#Visualization
plt.imshow(fill, cmap = plt.cm.gray)
plt.axis('off')

#Skeletonize Filter
skeleton = skeletonize(invert(fill),method='lee')
im = invert(skeleton)

#Visualization
plt.imshow(im, cmap = plt.cm.gray)
#plt.axis('off')
#plt.savefig('Mask3.eps', bbox_inches='tight', pad_inches=0)
plt.show()
