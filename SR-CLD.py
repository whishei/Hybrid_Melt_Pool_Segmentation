import numpy as np
import skimage
from PIL import Image
from skimage.feature import canny, blob_dog, peak_local_max

from skimage.util import invert
from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import skeletonize, thin, dilation, disk,closing
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import random_walker

from skimage.filters.rank import median 
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import seaborn as sns

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

#import cv2 as cv
import scipy.ndimage as ndimage  
import time

NASA = np.load('NASA_337_erosion.npy')
RF = np.load('crop337_RF.npy')

#Opening up the edited mask
mask_raw = Image.open('crop337_mask.png')
mask = mask_raw.convert('L')
truth_mask = np.asarray(mask)


#Creating the second training mask
truth_mask_interior = (truth_mask > 120)
truth_mask_boundary = (truth_mask < 102)
truth_mask_80 = (truth_mask <= 120)
truth_mask_75 = (truth_mask >= 102)
truth_mask_other = np.logical_and(truth_mask_80,truth_mask_75)

new_truth_mask = np.full(truth_mask.shape, -1)
new_truth_mask[truth_mask_other] = 1 
new_truth_mask[truth_mask_interior] = 2 
new_truth_mask[truth_mask_boundary] = 0


man_seg = new_truth_mask
print (man_seg.shape)

st = time.time()
#mult_k = 1
Ps = []
mult_k = 1
maxs = []
for k in range (0,int(len(NASA)/mult_k)):
    coords = []
    for i in range(0,mult_k):
        length = 0
        start_pix = 0
        for j in range(0,len(man_seg[i+(k*mult_k)])):
            if man_seg[i+(k*mult_k)][j] == 2: 
                length = length + 1
            else:
                if length == 0:
                    start_pix = start_pix + 1
                if length != 0: 
                    if start_pix != 0:
                        coords.append(length)
                    else: 
                        start_pix = start_pix + 1
                length = 0

    if coords != []:
       maxs.append(max(coords))
    vals, edges = np.histogram(coords,50,range = [0,450])


    middles = [] 
    num = []
    for i in range(0,len(edges)-1):
        middle = (edges[i]+ edges[i+1])/2 
        middles.append(middle)
        num.append(vals[i]*middle)
    
    
    den = sum(num)
    
    if den == 0:
        P = num 
    else:
        P = num/den   
    Ps.append(P)

    
Ps = np.array(Ps)
end = time.time()
execution = end - st
print (execution)

print (middles[:25])#*0.343)
print (np.unique(maxs))
#a = np.random.random((16, 16))

#NASA HZ Distribution = 192.73896884918213
#RF HZ Distribution = 188.45563912391663

#np.save('RF_Horizontal_Distribution.npy',Ps)

#Ps = np.load('NASA_Horizontal_Distribution.npy')

#print (Ps.shape)
new_Ps = Ps[:,:25]
#Visualization 
fig, ax = plt.subplots(figsize=(1, 5))

my_cmap = plt.cm.get_cmap('Spectral')
my_cmap = my_cmap.reversed()
my_cmap.set_over("white")
my_cmap.set_under("white")
plot_1 = sns.heatmap(new_Ps,cmap = my_cmap,vmax = 0.35,yticklabels=False,xticklabels=False) #cmap = my_cmap
#plot_1.set_xlabel('X-Axis', fontsize=10)
plot_1.set_ylabel('P', rotation = 0, fontsize=10)
#plt.savefig('Truth_HZ_SR_337.png', bbox_inches='tight', pad_inches=0)
plt.show()


#Vertical Distribution
#man_seg  = Test1
man_seg = man_seg.T
mult_k = 1
st = time.time()
#mult_k = 1
Ps = []
mult_k = 1
maxs = []
for k in range (0,int(len(NASA)/mult_k)):
    coords = []
    for i in range(0,mult_k):
        length = 0
        start_pix = 0
        for j in range(0,len(man_seg[i+(k*mult_k)])):
            if man_seg[i+(k*mult_k)][j] == 2: 
                length = length + 1
            else:
                if length == 0:
                    start_pix = start_pix + 1
                if length != 0: 
                    if start_pix != 0:
                        coords.append(length)
                    else: 
                        start_pix = start_pix + 1
                length = 0
                
    if coords != []:
       maxs.append(max(coords))
    vals, edges = np.histogram(coords,25,range = [0,240])
    
    middles = []
    num = []
    for i in range(0,len(edges)-1):
        middle = (edges[i]+ edges[i+1])/2 
        middles.append(middle)
        num.append(vals[i]*middle)
    
    
    den = sum(num)

    if den == 0:
        P = num 
    else:
        P = num/den  
    Ps.append(P)
    

Ps = np.array(Ps)
Ps = Ps.T
end = time.time()
execution = end - st
print (execution)
print (middles[:10])
print (np.unique(maxs))
#a = np.random.random((16, 16))

#NASA VT Distribution = 197.434907913208
#RF VT Distribution = 195.33019709587097

#np.save('NASA_Vertical_Distribution.npy',Ps)

#a = np.random.random((16, 16))
#np.save('Vertical_Distribution.npy',Ps)
#
#Ps = np.load('RF_Vertical_Distribution.npy')

print (Ps.shape)
new_Ps = Ps[:10,:]
#print (middles[:60]*0.343)


fig, ax = plt.subplots(figsize=(5, 1))

#create heatmap
#sns.heatmap(data, linewidths=.3)
my_cmap = plt.cm.get_cmap('Spectral')
my_cmap = my_cmap.reversed()
my_cmap.set_over("white")
my_cmap.set_under("white")
plot_1 = sns.heatmap(new_Ps,cmap = my_cmap,vmax = 0.5, xticklabels=False,yticklabels=False)#, vmax = 0.45)
#plot_1.set_xlabel('X-Axis', fontsize=10)
plot_1.set_xlabel('P', fontsize=10)
plt.savefig('Truth_VT_SR_337.png', bbox_inches='tight', pad_inches=0)
plt.show()

#print (coords)

