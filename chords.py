#UTEP Project
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
import seaborn as sns

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import cv2 as cv
import scipy.ndimage as ndimage  


#Opening the mask of interest
image = np.load('Results\crop45_seg.npy')
man_seg = image
#Visualizing the mask
plt.imshow(image,cmap = plt.cm.gray)
plt.colorbar()
plt.show()



mult_k = 100
Ps = []
for k in range (0,108):
    coords = []
    for i in range(0,mult_k):
        length = 0
        for j in range(0,len(man_seg[i+(k*mult_k)])):
            if man_seg[i+(k*mult_k)][j] == 2: #= man_seg[i][j+1]:
                length = length + 1
                if j == len(man_seg[i+(k*mult_k)]) - 1:
                    coords.append(length)
            else:
                if length != 0:
                    coords.append(length)
                length = 0
        #print (i + (k*50))
    vals, edges = np.histogram(coords,50,range = [0,50])
    
    #middle = []
    
    num = []
    for i in range(0,len(edges)-1):
        middle = (edges[i]+ edges[i+1])/2 
        num.append(vals[i]*middle)
    
    
    den = sum(num)
    
    if den == 0:
        P = num 
    else:
        P = num/den   
    Ps.append(P)
    
    #Ps.append(vals)
    #print (k)
    
Ps = np.array(Ps)
#a = np.random.random((16, 16))


#np.save('Horizontal_Distribution.npy',Ps)

fig, ax = plt.subplots(figsize=(1, 5))

#create heatmap
#sns.heatmap(data, linewidths=.3)
#sns.color_palette("Spectral", as_cmap=True)

my_cmap = plt.cm.get_cmap('Spectral')
my_cmap = my_cmap.reversed()
my_cmap.set_over("white")
my_cmap.set_under("white")
plot_1 = sns.heatmap(Ps, yticklabels=False, xticklabels=False, cmap = my_cmap,vmax = 0.06, vmin = 0.001)
#plot_1.set_xlabel('X-Axis', fontsize=10)
plot_1.set_ylabel('P', rotation = 0, fontsize=10)
plt.savefig('Heatmap_new_new.png', bbox_inches='tight', pad_inches=0)
plt.show()
#fig = plot_1.get_figure()
#fig.savefig("out.png", figsize = (1,5))
my_dpi = 3600

'''
plt.figure(dpi = my_dpi, figsize = (1,5))

#plt.imshow(Ps, cmap='hot', interpolation='nearest')
#plt.colorbar()
plt.savefig('Figures/Heatmap.png', bbox_inches='tight', pad_inches=0)
plt.show()
#ax = sns.heatmap(Ps, linewidth=0.5)
#plt.show()

#Vertical Coord Lengths
man_seg = man_seg.T
mult_k = 100
#plt.imshow(man_seg,cmap=plt.cm.gray)
#plt.colorbar()
#plt.axis('off')
#plt.savefig('Look.eps', bbox_inches='tight', pad_inches=0)
plt.show()

#print (man_seg)
Ps = []
for k in range (0,108):
    coords = []
    for i in range(0,mult_k):
        length = 0
        for j in range(0,len(man_seg[i+(k*mult_k)])):
            if man_seg[i+(k*mult_k)][j] == 2: #== man_seg[i][j+1]:
                length = length + 1
                if j == len(man_seg[i+(k*mult_k)]) - 1:
                    coords.append(length)
            else:
                if length != 0:
                    coords.append(length)
                length = 0
    #print (coords)
    vals, edges = np.histogram(coords,25,range = [0,25])
    
    #Ps.append(vals)
    
    #middle = []
    num = []
    for i in range(0,len(edges)-1):
        middle = (edges[i]+ edges[i+1])/2 
        num.append(vals[i]*middle)
    
    
    den = sum(num)
    #print (vals)
    if den == 0:
        P = num 
    else:
        P = num/den  
    Ps.append(P)
    #print (k)
    

Ps = np.array(Ps)
Ps = Ps.T
#a = np.random.random((16, 16))
np.save('Vertical_Distribution.npy',Ps)
fig, ax = plt.subplots(figsize=(5, 1))

#create heatmap
#sns.heatmap(data, linewidths=.3)
my_cmap = plt.cm.get_cmap('Spectral')
my_cmap = my_cmap.reversed()
my_cmap.set_over("white")
my_cmap.set_under("white")
plot_1 = sns.heatmap(Ps, yticklabels=False, xticklabels=False, cmap = my_cmap,vmax = 0.1, vmin = 0.001)
#plot_1.set_xlabel('X-Axis', fontsize=10)
plot_1.set_xlabel('P', fontsize=10)
plt.savefig('Heatmap_new_new2.png', bbox_inches='tight', pad_inches=0)
plt.show()
#fig = plot_1.get_figure()
#fig.savefig("out.png", figsize = (1,5))
my_dpi = 3600

plt.figure(dpi = my_dpi, figsize = (1,5))

#plt.imshow(Ps, cmap='hot', interpolation='nearest')
#plt.colorbar()
plt.savefig('Heatmap.png', bbox_inches='tight', pad_inches=0)
plt.show()
#ax = sns.heatmap(Ps, linewidth=0.5)
#plt.show()

'''

'''
plot_2 = sns.heatmap(Ps,yticklabels=False, xticklabels=False)
fig = plot_2.get_figure()
fig.savefig("out2.png", figsize = (5,1))

my_dpi = 3600
plt.figure(dpi = my_dpi, figsize = (5,1))

#plt.imshow(Ps, cmap='hot', interpolation='nearest')
#plt.colorbar()
plt.savefig('Heatmap2.png', bbox_inches='tight', pad_inches=0)
plt.show()
#ax = sns.heatmap(Ps, linewidth=0.5)
#plt.show()

my_dpi = 3600

plt.figure(dpi = my_dpi, figsize = (5,1))
plt.imshow(Ps, cmap='hot', interpolation='nearest')
#plt.colorbar()
plt.savefig('Heatmap2.png', bbox_inches='tight', pad_inches=0)
plt.show()
'''

'''
plt.title('Horizontal Coord Lengths of Edited')
plt.hist(coords,250, range=[0,250])
#plt.savefig('Bottom_edited_horizontal_histogram.png', bbox_inches='tight', pad_inches=0)
plt.show()

plt.title('Horizontal Coord Lengths of Edited')
plt.plot(P)
#plt.savefig('Bottom_edited_horizontal_histogram.png', bbox_inches='tight', pad_inches=0)
plt.show()


print ('Done1')
'''
'''
#Horizontal Coord Lengths
man_seg = new_image1_mask
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

vals, edges = np.histogram(coords,50,range = [0,50])

#middle = []
num = []
for i in range(0,len(edges)-1):
    middle = (edges[i]+ edges[i+1])/2 
    num.append(vals[i]*middle)


den = sum(num)

P = num/den  

print ('Done1')
plt.title('Horizontal Coord Lengths of Slice 1')
#plt.hist(coords,250, range=[0,250])
plt.plot(P)
plt.savefig('Horizontal_Coord_Lengths_of_Slice_1.png', bbox_inches='tight', pad_inches=0)
plt.show()
#plt.savefig('Bottom_edited_horizontal_histogram.png', bbox_inches='tight', pad_inches=0)
#plt.show()

edit_hz = coords


#Vertical Coord Lengths
man_seg = man_seg.T
#print (man_seg)
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

vals, edges = np.histogram(coords,25,range = [0,25])

#middle = []
num = []
for i in range(0,len(edges)-1):
    middle = (edges[i]+ edges[i+1])/2 
    num.append(vals[i]*middle)


den = sum(num)

P = num/den  

plt.title('Vertical Coord Lengths of Slice 1')
#plt.hist(coords,250, range=[0,250])
plt.plot(P)
plt.savefig('Vertical_Coord_Lengths_of_SLice_1.png', bbox_inches='tight', pad_inches=0)
plt.show()
     
print ('Done2')
#plt.title('Vertical Coord Lengths of Edited')
#plt.hist(coords,50,range=[0, 50])
#plt.savefig('Bottom_edited_vertical_histogram.png', bbox_inches='tight', pad_inches=0)
#plt.show()
'''
'''
edit_v = coords


#Horizontal Coord Lengths
man_seg = new_image2_mask
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

#plt.title('Horizontal Coord Lengths of RF')
#plt.hist(coords,250,range=[0, 250])
#plt.savefig('Bottom_horizontal_histogram.png', bbox_inches='tight', pad_inches=0)
#plt.show()

vals, edges = np.histogram(coords,250,range = [0,250])

#middle = []
num = []
for i in range(0,len(edges)-1):
    middle = (edges[i]+ edges[i+1])/2 
    num.append(vals[i]*middle)


den = sum(num)

P = num/den  

plt.title('Horizontal Coord Lengths of NotEdited')
#plt.hist(coords,250, range=[0,250])
plt.plot(P)
plt.savefig('Test1_NotEdited_Histogram_HZ.png', bbox_inches='tight', pad_inches=0)
plt.show()

nonedit_hz = coords

print ('Done3')

#Vertical Coord Lengths
man_seg = man_seg.T
#print (man_seg)
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
            
#plt.title('Vertical Coord Lengths of RF')
#plt.hist(coords,50, range=[0, 50])
#plt.savefig('Bottom_vertical_histogram.png', bbox_inches='tight', pad_inches=0)
#plt.show()

nonedit_v = coords
vals, edges = np.histogram(coords,50,range = [0,50])

#middle = []
num = []
for i in range(0,len(edges)-1):
    middle = (edges[i]+ edges[i+1])/2 
    num.append(vals[i]*middle)


den = sum(num)

P = num/den  

#print ('Done1')
plt.title('Vertical Coord Lengths of NotEdited')
#plt.hist(coords,250, range=[0,250])
plt.plot(P)
plt.savefig('Test1_NotEdited_Histogram_V.png', bbox_inches='tight', pad_inches=0)
plt.show()
'''
'''
print ('Done4')
#Vertical Histograms
fig, ax = plt.subplots(1,2,figsize=(12, 8))
plt.subplot(121)
plot1=plt.hist(nonedit_v,50,range = [0,50])
plt.title("Not Edited Vertical Chord Length")
plt.subplot(122)
plot2=plt.hist(edit_v,50,range = [0,50])
plt.title("Edited Vertical Chord Length")
#plt.savefig('Test1_vertical_histogram_2.png', bbox_inches='tight', pad_inches=0)
plt.show()


diff=plt.bar(np.arange(50), 
             height=(plot1[0]-plot2[0])) 
plt.title("Not Edited - Edited Vertical Chord Length")
#plt.savefig('Test1_vertical_change_histogram.png', bbox_inches='tight', pad_inches=0)
plt.show()


#Vertical Histograms
fig, ax = plt.subplots(1,2,figsize=(12, 8))
plt.subplot(121)
plot1=plt.hist(nonedit_hz,250,range = [0,250])
plt.title("Not Edited Horizontal Chord Length")
plt.subplot(122)
plot2=plt.hist(edit_hz,250,range = [0,250])
plt.title("Edited Horizontal Chord Length")
#plt.savefig('Test1_horizontal_histogram_2.png', bbox_inches='tight', pad_inches=0)
plt.show()


diff=plt.bar(np.arange(250), 
             height=(plot1[0]-plot2[0])) 
plt.title("Not Edited - Edited Horizontal Chord Length")
#plt.savefig('Test1_horizontal_change_histogram.png', bbox_inches='tight', pad_inches=0)
plt.show()


#Horizontal Difference
plt.title('Horizontal Chord Length Difference')
plt.hist(nonedit_hz - edit_hz,250, range=[0, 250])
plt.savefig('Horizontal_Chord_Length_Difference.png', bbox_inches='tight', pad_inches=0)
plt.show()

#Vertical
plt.title('Vertical Chord Length Difference')
plt.hist(nonedit_v - edit_v,50, range=[0, 50])
plt.savefig('Vertical_Chord_Length_Difference.png', bbox_inches='tight', pad_inches=0)
plt.show()
'''