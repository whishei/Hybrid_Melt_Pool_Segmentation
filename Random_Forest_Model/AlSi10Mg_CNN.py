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
import time


#Opening the training data and the test data 

Image.MAX_IMAGE_PIXELS = 399279028
my_dpi = 3600

#Opening the first training image
train1_raw = Image.open('AlSi10Mg-Training/Training_1_crop1.png')
train1 = train1_raw.convert('L')
train1 = np.asarray(train1)

train1_raw_1 = Image.open('AlSi10Mg-Training/Training_1_crop1_rot90.png')
train1_1 = train1_raw_1.convert('L')
train1_1 = np.asarray(train1_1)

train1_raw_2 = Image.open('AlSi10Mg-Training/Training_1_crop1_rot180.png')
train1_2 = train1_raw_2.convert('L')
train1_2 = np.asarray(train1_2)

#Opening the second training image
train2_raw = Image.open('AlSi10Mg-Training/Training_2_crop88.tif')
train2 = train2_raw.convert('L')
train2 = np.asarray(train2)

train2_raw_1 = Image.open('AlSi10Mg-Training/Training_2_crop88_rot90.tif')
train2_1 = train2_raw_1.convert('L')
train2_1 = np.asarray(train2_1)

train2_raw_2 = Image.open('AlSi10Mg-Training/Training_2_crop88_rot180.tif')
train2_2 = train2_raw_2.convert('L')
train2_2 = np.asarray(train2_2)

#Opening the third training image
train3_raw = Image.open('AlSi10Mg-Training/Training_3_difficult.tif')
train3 = train3_raw.convert('L')
train3 = np.asarray(train3)

train3_raw_1 = Image.open('AlSi10Mg-Training/Training_3_difficult_rot90.tif')
train3_1 = train3_raw_1.convert('L')
train3_1 = np.asarray(train3_1)

train3_raw_2 = Image.open('AlSi10Mg-Training/Training_3_difficult_rot180.tif')
train3_2 = train3_raw_2.convert('L')
train3_2 = np.asarray(train3_2)

#Opening the fourth training image
train4_raw = Image.open('AlSi10Mg-Training/Training_4_crop393.tif')
train4 = train4_raw.convert('L')
train4 = np.asarray(train4)

train4_raw_1 = Image.open('AlSi10Mg-Training/Training_4_crop393_rot90.tif')
train4_1 = train4_raw_1.convert('L')
train4_1 = np.asarray(train4_1)

train4_raw_2 = Image.open('AlSi10Mg-Training/Training_4_crop393_rot180.tif')
train4_2 = train4_raw_2.convert('L')
train4_2 = np.asarray(train4_2)

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
train1_mask = np.load('AlSi10Mg-Training/Training_1_mask.npy')
train1_1_mask = np.load('AlSi10Mg-Training/Training_1_mask_90.npy')
train1_2_mask = np.load('AlSi10Mg-Training/Training_1_mask_180.npy')

#Opening the second training mask
train2_mask = np.load('AlSi10Mg-Training/Training_2_mask.npy')
train2_1_mask = np.load('AlSi10Mg-Training/Training_2_mask_90.npy')
train2_2_mask = np.load('AlSi10Mg-Training/Training_2_mask_180.npy')

#Opening the third training mask
train3_mask = np.load('AlSi10Mg-Training/Training_3_mask.npy')
train3_1_mask = np.load('AlSi10Mg-Training/Training_3_mask_90.npy')
train3_2_mask = np.load('AlSi10Mg-Training/Training_3_mask_180.npy')

#Opening the fourth training mask
train4_mask = np.load('AlSi10Mg-Training/Training_4_mask.npy')
train4_1_mask = np.load('AlSi10Mg-Training/Training_4_mask_90.npy')
train4_2_mask = np.load('AlSi10Mg-Training/Training_4_mask_180.npy')

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

'''
#Visualizing test image
plt.imshow(test_image,cmap=plt.cm.gray)
plt.colorbar()
plt.show()
'''

#print ("Check1")

#############Random Forest Classifier

sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)

#Features for the two training images
train1_features = features_func(train1)
train1_1_features = features_func(train1_1)
train1_2_features = features_func(train1_2)
train2_features = features_func(train2)
train2_1_features = features_func(train2_1)
train2_2_features = features_func(train2_2)
train3_features = features_func(train3)
train3_1_features = features_func(train3_1)
train3_2_features = features_func(train3_2)
train4_features = features_func(train4)
train4_1_features = features_func(train4_1)
train4_2_features = features_func(train4_2)


clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=10, max_samples=0.05)

#Reshaping from mXm array to (mxm)x1 array
train1_mask = train1_mask.reshape(-1)
train1_1_mask = train1_1_mask.reshape(-1)
train1_2_mask = train1_2_mask.reshape(-1)
train2_mask = train2_mask.reshape(-1)
train2_1_mask = train2_1_mask.reshape(-1)
train2_2_mask = train2_2_mask.reshape(-1)
train3_mask = train3_mask.reshape(-1)
train3_1_mask = train3_1_mask.reshape(-1)
train3_2_mask = train3_2_mask.reshape(-1)
train4_mask = train4_mask.reshape(-1)
train4_1_mask = train4_1_mask.reshape(-1)
train4_2_mask = train4_2_mask.reshape(-1)

#Reshaping from mxmx15 to (mxm)x15 array
train1_features_ = np.reshape(train1_features, (-1,train1_features.shape[2]))
train1_1_features_ = np.reshape(train1_1_features, (-1,train1_1_features.shape[2]))
train1_2_features_ = np.reshape(train1_2_features, (-1,train1_2_features.shape[2]))
train2_features_ = np.reshape(train2_features, (-1,train2_features.shape[2]))
train2_1_features_ = np.reshape(train2_1_features, (-1,train2_1_features.shape[2]))
train2_2_features_ = np.reshape(train2_2_features, (-1,train2_2_features.shape[2]))
train3_features_ = np.reshape(train3_features, (-1,train3_features.shape[2]))
train3_1_features_ = np.reshape(train3_1_features, (-1,train3_1_features.shape[2]))
train3_2_features_ = np.reshape(train3_2_features, (-1,train3_2_features.shape[2]))
train4_features_ = np.reshape(train4_features, (-1,train4_features.shape[2]))
train4_1_features_ = np.reshape(train4_1_features, (-1,train4_1_features.shape[2]))
train4_2_features_ = np.reshape(train4_2_features, (-1,train4_2_features.shape[2]))

#Choosing only the known and labeled pixels
ind1 = (train1_mask > -1)
ind1_1 = (train1_1_mask > -1)
ind1_2 = (train1_2_mask > -1)
ind2 = (train2_mask > -1)
ind2_1 = (train2_1_mask > -1)
ind2_2 = (train2_2_mask > -1)
ind3 = (train3_mask > -1)
ind3_1 = (train3_1_mask > -1)
ind3_2 = (train3_2_mask > -1)
ind4 = (train4_mask > -1)
ind4_1 = (train4_1_mask > -1)
ind4_2 = (train4_2_mask > -1)

#Choosing only the unknown and unlabeled pixels
ind1_ = (train1_mask == -1)
ind1_1_ = (train1_1_mask == -1)
ind1_2_ = (train1_2_mask == -1)
ind2_ = (train2_mask == -1)
ind2_1_ = (train2_1_mask == -1)
ind2_2_ = (train2_2_mask == -1)
ind3_ = (train3_mask == -1)
ind3_1_ = (train3_1_mask == -1)
ind3_2_ = (train3_2_mask == -1)
ind4_ = (train4_mask == -1)
ind4_1_ = (train4_1_mask == -1)
ind4_2_ = (train4_2_mask == -1)

#Creating the training data 
X_train1 = train1_features_[ind1,:]
X_train1_1 = train1_1_features_[ind1_1,:]
X_train1_2 = train1_2_features_[ind1_2,:]
X_train2 = train2_features_[ind2,:]
X_train2_1 = train2_1_features_[ind2_1,:]
X_train2_2 = train2_2_features_[ind2_2,:]
X_train3 = train3_features_[ind3,:]
X_train3_1 = train3_1_features_[ind3_1,:]
X_train3_2 = train3_2_features_[ind3_2,:]
X_train4 = train4_features_[ind4,:]
X_train4_1 = train4_1_features_[ind4_1,:]
X_train4_2 = train4_2_features_[ind4_2,:]
y_train1 = train1_mask[ind1]
y_train1_1 = train1_1_mask[ind1_1]
y_train1_2 = train1_2_mask[ind1_2]
y_train2 = train2_mask[ind2]
y_train2_1 = train2_1_mask[ind2_1]
y_train2_2 = train2_2_mask[ind2_2]
y_train3 = train3_mask[ind3]
y_train3_1 = train3_1_mask[ind3_1]
y_train3_2 = train3_2_mask[ind3_2]
y_train4 = train4_mask[ind4]
y_train4_1 = train4_1_mask[ind4_1]
y_train4_2 = train4_2_mask[ind4_2]
X_train = np.append(X_train1,X_train1_1,axis = 0)
X_train = np.append(X_train,X_train1_2,axis = 0)
X_train = np.append(X_train,X_train2,axis = 0)
X_train = np.append(X_train,X_train2_1,axis = 0)
X_train = np.append(X_train,X_train2_2,axis = 0)
X_train = np.append(X_train,X_train3,axis = 0)
X_train = np.append(X_train,X_train3_1,axis = 0)
X_train = np.append(X_train,X_train3_2,axis = 0)
X_train = np.append(X_train,X_train4,axis = 0)
X_train = np.append(X_train,X_train4_1,axis = 0)
X_train = np.append(X_train,X_train4_2,axis = 0)

y_train = np.append(y_train1,y_train1_1)
y_train = np.append(y_train,y_train1_2)
y_train = np.append(y_train,y_train2)
y_train = np.append(y_train,y_train2_1)
y_train = np.append(y_train,y_train2_2)
y_train = np.append(y_train,y_train3)
y_train = np.append(y_train,y_train3_1)
y_train = np.append(y_train,y_train3_2)
y_train = np.append(y_train,y_train4)
y_train = np.append(y_train,y_train4_1)
y_train = np.append(y_train,y_train4_2)

#print ("Check2")
clf.fit(X_train,y_train)
#print ("Check3")

#Predicting for our test image
test_features = features_func(test_image)
TEST_image = np.reshape(test_features, (-1,test_features.shape[2]))
X_test = TEST_image 

#Timing
st = time.time()
y_test = clf.predict(X_test)
end = time.time()
execution = end - st
print (execution)
#print ("Check4")


#Reshaping back into orginal image shape
remake_img = np.reshape(y_test, test_image.shape)

#Unique classifications (0 - boundary, 1 - background, 2 - interior)
values = np.unique(remake_img)

'''
#Visualization
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(test_image,cmap=plt.cm.gray)
ax[0].set_title('Raw Image')
ax[1].imshow(remake_img, cmap=plt.cm.gray)
ax[1].set_title('Segmentation of crop45.tif')
fig.tight_layout()
#plt.savefig('Results\crop45_seg1.png', bbox_inches='tight', pad_inches=0)
plt.show()
'''

#Saving segmentation matrix
#np.save('crop45_melts.npy',remake_img)

'''
#Visualization_specific
plt.figure(dpi = my_dpi)
plt.imshow(remake_img,cmap=plt.cm.gray)
plt.axis('off')
#plt.savefig('Results\crop45_seg2.png', bbox_inches='tight', pad_inches=0)
plt.show()
'''
