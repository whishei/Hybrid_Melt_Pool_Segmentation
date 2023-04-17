#UTEP Project
import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from functools import partial

from skimage.feature import canny, blob_dog, peak_local_max
from skimage.util import invert
from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import skeletonize, thin, dilation, disk,closing
from skimage.segmentation import random_walker
from skimage.filters.rank import median 
from skimage import segmentation, feature, future


#Opening the training data and the test data 

Image.MAX_IMAGE_PIXELS = 399279028
my_dpi = 3600

#Opening the first training image
train1_raw = Image.open('Training/Training_1_crop1.png')
train1 = train1_raw.convert('L')
train1 = np.asarray(train1)

#Opening the second training image
train2_raw = Image.open('Training/Training_2_crop88.tif')
train2 = train2_raw.convert('L')
train2 = np.asarray(train2)

#Opening the third training image
train3_raw = Image.open('Training/Training_3_difficult.tif')
train3 = train3_raw.convert('L')
train3 = np.asarray(train3)

#Opening the fourth training image
train4_raw = Image.open('Training/Training_4_crop393.tif')
train4 = train4_raw.convert('L')
train4 = np.asarray(train4)

'''
#Visualizing training images
plt.imshow(train1,cmap=plt.cm.gray)
plt.show()
plt.imshow(train2,cmap=plt.cm.gray)
plt.show()
plt.imshow(train3,cmap=plt.cm.gray)
plt.show()
plt.imshow(train4,cmap=plt.cm.gray)
plt.show()
'''

#Opening the first training mask
train1_mask = np.load('Training/Training_1_mask.npy')

#Opening the second training mask
train2_mask = np.load('Training/Training_2_mask.npy')

#Opening the third training mask
train3_mask = np.load('Training/Training_3_mask.npy')

#Opening the fourth training mask
train4_mask = np.load('Training/Training_4_mask.npy')

'''
#Visualizing training masks
plt.imshow(train1_mask,cmap=plt.cm.gray)
plt.colorbar()
plt.show()
plt.imshow(train2_mask,cmap=plt.cm.gray)
plt.colorbar()
plt.show()
plt.imshow(train3_mask,cmap=plt.cm.gray)
plt.colorbar()
plt.show()
plt.imshow(train4_mask,cmap=plt.cm.gray)
plt.colorbar()
plt.show()
'''


#Opening the test image
test_image_raw = Image.open('Testing/crop45.tif')
test_image = test_image_raw.convert('L')
test_image = np.asarray(test_image)

#Visualizing test image
plt.imshow(test_image,cmap=plt.cm.gray)
plt.colorbar()
plt.show()

#############Random Forest Classifier

sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)

#Features for the two training images
train1_features = features_func(train1)
train2_features = features_func(train2)
train3_features = features_func(train3)
train4_features = features_func(train4)


clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=10, max_samples=0.05)

#Reshaping from mXm array to (mxm)x1 array
train1_mask = train1_mask.reshape(-1) 
train2_mask = train2_mask.reshape(-1) 
train3_mask = train3_mask.reshape(-1)
train4_mask = train4_mask.reshape(-1)

#Reshaping from mxmx15 to (mxm)x15 array
train1_features_ = np.reshape(train1_features, (-1,train1_features.shape[2]))
train2_features_ = np.reshape(train2_features, (-1,train2_features.shape[2]))
train3_features_ = np.reshape(train3_features, (-1,train3_features.shape[2]))
train4_features_ = np.reshape(train4_features, (-1,train4_features.shape[2]))

#Choosing only the known and labeled pixels
ind1 = (train1_mask > -1)
ind2 = (train2_mask > -1)
ind3 = (train3_mask > -1)
ind4 = (train4_mask > -1)
#Choosing only the unknown and unlabeled pixels
ind1_ = (train1_mask == -1)
ind2_ = (train2_mask == -1)
ind3_ = (train3_mask == -1)
ind4_ = (train4_mask == -1)

#Creating the training data 
X_train1 = train1_features_[ind1,:]
X_train2 = train2_features_[ind2,:]
X_train3 = train3_features_[ind3,:]
X_train4 = train4_features_[ind4,:]
y_train1 = train1_mask[ind1]
y_train2 = train2_mask[ind2]
y_train3 = train3_mask[ind3]
y_train4 = train4_mask[ind4]
X_train = np.append(X_train1,X_train2,axis = 0)
X_train = np.append(X_train,X_train3,axis = 0)
X_train = np.append(X_train,X_train4,axis = 0)
y_train = np.append(y_train1,y_train2)
y_train = np.append(y_train,y_train3)
y_train = np.append(y_train,y_train4)


clf.fit(X_train,y_train)

#Predicting for our test image
test_features = features_func(test_image)
TEST_image = np.reshape(test_features, (-1,test_features.shape[2]))
X_test = TEST_image 
y_test = clf.predict(X_test)


#Reshaping back into orginal image shape
remake_img = np.reshape(y_test, test_image.shape)

#Unique classifications (0 - boundary, 1 - background, 2 - interior)
values = np.unique(remake_img)


#Visualization
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(test_image,cmap=plt.cm.gray)
ax[0].set_title('Raw Image')
ax[1].imshow(remake_img, cmap=plt.cm.gray)
ax[1].set_title('Segmentation of crop45.tif')
fig.tight_layout()
#plt.savefig('Results\crop45_seg1.png', bbox_inches='tight', pad_inches=0)
plt.show()

np.save('Results\crop45_seg.npy',remake_img)
'''
#Visualization_specific
plt.figure(dpi = my_dpi)
plt.imshow(remake_img,cmap=plt.cm.gray)
plt.axis('off')
#plt.savefig('Results\crop45_seg2.png', bbox_inches='tight', pad_inches=0)
plt.show()
'''
