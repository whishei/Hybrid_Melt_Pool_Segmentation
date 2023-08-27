#UTEP Project
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

#########  Opening Up Segmentation Results

#Opening the segmentation results
NASA = np.load('NASA_337_erosion.npy')
RF = np.load('crop337_RF.npy')

#Opening up the edited mask
mask_raw = Image.open('crop337_mask.png')
mask = mask_raw.convert('L')
crop88_mask = np.asarray(mask)

#Creating the second training mask
crop88_mask_interior = (crop88_mask > 120)
crop88_mask_boundary = (crop88_mask < 102)
crop88_mask_80 = (crop88_mask <= 120)
crop88_mask_75 = (crop88_mask >= 102)
crop88_mask_other = np.logical_and(crop88_mask_80,crop88_mask_75)

new_crop88_mask = np.full(crop88_mask.shape, -1)
new_crop88_mask[crop88_mask_other] = 1 #119
new_crop88_mask[crop88_mask_interior] = 2 #255
new_crop88_mask[crop88_mask_boundary] = 0


########### Calculating Jaccards Index

'''
ground_truth = np.zeros((1000,1000))
ground_truth[new_crop88_mask == 0] = 1

NASA_model = np.zeros((1000,1000))
NASA_model[NASA == 0] = 1

RF_model = np.zeros((1000,1000))
RF_model[RF == 0] = 1


def Jaccards(ground_truth, computer_made):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(0,len(ground_truth)):
        for j in range(0,len(ground_truth[i])):
            if ground_truth[i][j] == 1 and computer_made[i][j] == 1:
                TP = TP + 1
            elif ground_truth[i][j] == 0 and computer_made[i][j] == 1:
                FP = FP + 1
            elif ground_truth[i][j] == 1 and computer_made[i][j] == 0:
                FN = FN + 1
            elif ground_truth[i][j] == 0 and computer_made[i][j] == 0:
                TN = TN + 1
 
    return TP / (TP + FN + FP)


print (Jaccards(ground_truth,NASA_model))
print (Jaccards(ground_truth,RF_model))
'''
########### Saving Horizontal and Vertical Coords of entire image

'''
#Horizontal Coord Lengths

man_seg = RF
coords = []
for i in range(0,len(man_seg)):
    length = 0
    for j in range(0,len(man_seg[i])):
        if man_seg[i][j] == 2: #== man_seg[i][j+1]:
            length = length + 1
            if j == len(man_seg[i]) - 1:
                coords.append(length)
        else:
            if length != 0:
                coords.append(length)
            length = 0

#np.save('RF_337_horizontal_chords.npy',coords)

#Vertical Coord Lengths

man_seg = man_seg.T
coords = []
for i in range(0,len(man_seg)):
    length = 0
    for j in range(0,len(man_seg[i])):
        if man_seg[i][j] == 2: #== man_seg[i][j+1]:
            length = length + 1
            if j == len(man_seg[i]) - 1:
                coords.append(length)
        else:
            if length != 0:
                coords.append(length)
            length = 0
            
#coords = np.save('RF_337_vertical_chords.npy',coords)
'''


########## Calculating Horizontal and Vertical CLDs of entire image (using max coord value)
'''
coords_RF = np.load('RF_45_vertical_chords.npy')
coords_NASA = np.load('NASA_45_vertical_chords.npy')
coords_truth = np.load('truth_45_vertical_chords.npy')

vals_truth, edges_truth = np.histogram(coords_truth,50,range = [0,1200])
vals_NASA, edges_NASA = np.histogram(coords_NASA,50,range = [0,1200])
vals_RF, edges_RF = np.histogram(coords_RF,50,range = [0,1200])

middles = []
num_truth = []
num_NASA = []
num_RF = []
for i in range(0,len(edges_truth)-1):
    middle = (edges_truth[i]+ edges_truth[i+1])/2
    middles.append(middle)
    num_truth.append(vals_truth[i]*middle)
    num_NASA.append(vals_NASA[i]*middle)
    num_RF.append(vals_RF[i]*middle)


den_truth = sum(num_truth)
den_NASA = sum(num_NASA)
den_RF = sum(num_RF)

truth = num_truth/den_truth
NASA = num_NASA/den_NASA
RF = num_RF/den_RF 

middles = np.array(middles)
middles = middles*0.343

#plt.title('Truth - NASA Horizontal CLD')
#plt.hist(coords,250, range=[0,250])
#COLORSS
#DF873E - orange - CNN
#69E374 - green - truth
#EA334B - red
#52B5F9 - blue - RF

fig, ax = plt.subplots()
#ax.plot(middles,truth,color = '#69E374',label = 'Ground Truth')
ax.plot(middles,NASA-truth,color = '#DF873E',label = 'CNN')
ax.plot(middles,RF-truth,color = '#52B5F9',label = 'RF')
#ax.axis('equal')
ax.set_ylim([-0.06,0.06])
ax.set_xlim([0,max(middles)])
leg = ax.legend();
#plt.savefig('Horizontal_Errors2.svg', bbox_inches='tight', pad_inches=0)
plt.show()


#Vertical CLDs
vals_truth, edges_truth = np.histogram(coords_truth,50,range = [0,800])
vals_NASA, edges_NASA = np.histogram(coords_NASA,50,range = [0,800])
vals_RF, edges_RF = np.histogram(coords_RF,50,range = [0,800])


middles = []
num_truth = []
num_NASA = []
num_RF = []
for i in range(0,len(edges_truth)-1):
    middle = (edges_truth[i]+ edges_truth[i+1])/2
    middles.append(middle)
    num_truth.append(vals_truth[i]*middle)
    num_NASA.append(vals_NASA[i]*middle)
    num_RF.append(vals_RF[i]*middle)


den_truth = sum(num_truth)
den_NASA = sum(num_NASA)
den_RF = sum(num_RF)

truth = num_truth/den_truth
NASA = num_NASA/den_NASA
RF = num_RF/den_RF 

middles = np.array(middles)
middles = middles*0.343

#plt.title('Truth - NASA Horizontal CLD')
#plt.hist(coords,250, range=[0,250])
#COLORSS
#DF873E - orange - CNN
#69E374 - green - truth
#EA334B - red
#52B5F9 - blue - RF

fig, ax = plt.subplots()
#ax.plot(middles,truth,color = '#69E374',label = 'Ground Truth')
ax.plot(middles,NASA - truth,color = '#DF873E',label = 'CNN')
ax.plot(middles,RF - truth,color = '#52B5F9',label = 'RF')
#ax.axis('equal')
ax.set_ylim([-0.03,0.03])
ax.set_xlim([0,max(middles)])
leg = ax.legend();
#plt.savefig('Vertical_Errors2.svg', bbox_inches='tight', pad_inches=0)
plt.show()

'''
