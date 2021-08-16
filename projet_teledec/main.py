import time
from glob import glob

import cv2
import rasterio
import json, re, itertools, os
from pyrasta.raster import Raster
import geopandas
from osgeo import ogr

from conv import jp2_to_tif
from PIL import TiffImagePlugin
import gdal
import numpy as np
import tifffile as tiff
from fototex.foto import Foto, FotoSector, FotoBatch
from skimage import io
from fototex.foto_tools import degrees_to_cardinal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2 as cv
from skimage import feature

from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

#jp2_to_tif()

#foto = Foto("/home/paul/PycharmProjects/projet_teledec/data/T54SUE_20210427T012651_B03.jp2", band=None, method="block",
          #in_memory=True, data_chunk_size=50000)
#foto.run(window_size=19,keep_dc_component=False)

#foto.save_rgb()



#img = cv.imread('/home/paul/PycharmProjects/projet_teledec/pan_image_test.tif',
 #               cv.IMREAD_LOAD_GDAL | cv.IMREAD_COLOR)
#img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
from clustering import clustering, elbow, filtering

dataset = gdal.Open('/home/paul/PycharmProjects/projet_teledec/data/FOTO_method=block_wsize=13_dc=False_image=T19GDL_20210422T141731_B03_rgb.tif')



### Fetching the channels ###
band1 = dataset.GetRasterBand(1)

band2 = dataset.GetRasterBand(2)
band3 = dataset.GetRasterBand(3)
b1 = band1.ReadAsArray()
b2 = band2.ReadAsArray()
b3 = band3.ReadAsArray()

'''gtiff_driver = gdal.GetDriverByName('GTiff')
out_ds = gtiff_driver.Create('nat_color.tif',band1.XSize,band1.YSize, 3, band1.DataType)
out_ds.SetProjection()'''
### Display the arrays
img = np.dstack((b1,b2,b3))
#plt.figure(1°
#plt.imshow(img)
#plt.savefig('Tiff.png')
#plt.show()



### MASKS ###
## def mask(img)
'''lower_red = np.array([4,2,2])
upper_red = np.array([255,255,255])
mask = cv.inRange(img, lower_red, upper_red)
res = cv.bitwise_and(img,img,mask=mask)
plt.figure(2)
plt.imshow(mask)
plt.figure(3)
plt.imshow(res)
plt.show()'''
#plt.show()
#cv.imshow('output_FOTO',img)
#cv.waitKey(0)
#cv.destroyAllWindows()



####  FILTERING ####

ker = (13,13)




img = filtering(ker,'Gaussian',img)

#gray = cv.cvtColor(med_3, cv.COLOR_BGR2GRAY)


#### K-MEANS/CLUSTERING


'''
connectivity = kneighbors_graph(pixels,7)
K = 5                  ## nb of clusters'''

pixels = img.reshape((-1,3))
pixels = np.float32(pixels)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 70, 0.1)



#### ELBOW METHOD ###
#elbow(pixels,criteria)





labels, centers = clustering(img,'k_means',5)
#ward = clustering(gauss_17,'hierarchical',6)
#ward_labels = np.reshape(ward.labels_, (gauss_17.shape[0], gauss_17.shape[1]))
K=5
all_labels = []
all_centers = []
seg_imgs = []

### Loop on K ###

'''for k in range(3,7):
    labels, centers = clustering(img, 'k_means', k)
    all_labels.append(labels)
    all_centers.append(centers)


for i in range(len(all_centers)):
    all_labels[i] = all_labels[i].flatten()
    centers = all_centers[i]
    seg_imgs.append(centers[all_labels[i]])
    seg_imgs[i] = seg_imgs[i].reshape(img.shape)



for j in range(len(all_labels)):
    seg_img = seg_imgs[j]
    seg_img = seg_img.reshape(-1,3)
    labels = all_labels[j]
    seg_img[labels==j] = colors[0]
    seg_img = seg_img.reshape(img.shape)
    plt.figure()
    plt.imshow(seg_img)
    plt.title('K='+str(j+3))



fig,axs = plt.subplots(2,2)
axs[0,0].imshow(seg_imgs[0])
axs[0,0].set_title('K=3')
axs[0,0].legend()
axs[0,1].imshow(seg_imgs[1])
axs[0,1].set_title('K=4')
axs[1,0].imshow(seg_imgs[2])
axs[1,0].set_title('K=5')
axs[1,1].imshow(seg_imgs[3])
axs[1,1].set_title('K=6')'''





labels = labels.flatten()
#ward_labels = ward_labels.ravel()
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(img.shape)




#### CLUSTER ELIMINATION/DISCRIMINATION ####
### def cluster_discrimination(K,labels,img)

# masked_image = np.copy(img)
# masked_image = masked_image.reshape((-1,3))
# colors = np.random.rand(K,3)
#
# for i in range(K):
#     masked_image[labels==i] = colors[i]
#
#
# #masked_image[ward_labels==1] = [1,0,0]
# #masked_image[ward_labels==3] = [1,0,0]
# masked_image = masked_image.reshape(img.shape)
# #ward_labels = ward_labels.reshape(844,844)
#
# #### Legend ####
# ### def legend(labels,colors,K) à faire
# mycmap = plt.cm.jet
# values= np.unique(labels.ravel())
# #im = plt.imshow(masked_image)
# patches = [ mpatches.Patch(color = colors[i], label= "Cluster {c}".format(c=values[i]) ) for i in range(K)]
#





#### Display ####
### def display() à faire
'''fig, axs = plt.subplots(2, 2)
axs[0,0].imshow(img)
axs[0,0].set_title('Fototex Output')
axs[0,1].imshow(segmented_image)
axs[0,1].set_title('K_Means++ K='+str(K))
axs[1,0].imshow(masked_image)
axs[1,0].set_title('Cluster separation')
axs[1,0].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )'''
#axs[1,1].imshow(labels)
#axs[1,1].set_title('Hierarchical Clustering K='+str(K))
#plt.show()










### HISTOGRAMS

labels = labels.reshape(len(img[0]),len(img[1]))
foto_raster = Raster('/home/paul/PycharmProjects/projet_teledec/data/FOTO_method=block_wsize=13_dc=False_image=T19GDL_20210422T141731_B03_rgb.tif')
#def histo_cluster(labels,raster)



#foto_raster = Raster('/home/paul/PycharmProjects/projet_teledec/Kmeans.tif')

foto = foto_raster.read_array()
dict = {}
for i in range(K):
    dict[i] =  foto[:,labels==i]







fig, ax = plt.subplots(1, K)
for i in range(K):
    ax[i].hist(dict[i][0], bins='auto', color="red")
    ax[i].hist(dict[i][1], bins='auto', color="blue")
    ax[i].hist(dict[i][2], bins='auto', color="green")
    ax[i].set_title('Cluster ='+str(i))
    ax[i].axis([-50, 50, 0, 6000 ])
plt.show()

mean = [] ; min = [] ; max = []  ; median = [] ; std = []

for k in range(K):
    mean.append(np.mean(dict[k][0]))
    median.append(np.median(dict[k][0]))
    max.append(np.max(dict[k][0]))
    min.append(np.min(dict[k][0]))
    std.append(np.std(dict[k][0]))




'''urbain = []
other = []

for i in range(len(labels)):
    if labels[i] == 1 | labels[i] == 3:
        urbain.append(img[i])
    else:
        other.append(img[i])

urbain = np.array(urbain)
other = np.array(other)


#### STATS ON CLUSTERS ###
avgs = [np.mean(urbain[:,0]),np.mean(urbain[:,1]),np.mean(urbain[:,2])]
meds = [np.median(urbain[:,0]),np.median(urbain[:,1]),np.median(urbain[:,2])]
min = [np.min(urbain[:,0]),np.min(urbain[:,1]),np.min(urbain[:,2])]
max = [np.max(urbain[:,0]),np.max(urbain[:,1]),np.max(urbain[:,2])]
std = [np.std(urbain[:,0]),np.std(urbain[:,1]),np.std(urbain[:,2])]

avgs_ = [np.mean(other[:,0]),np.mean(other[:,1]),np.mean(other[:,2])]
meds_ = [np.median(other[:,0]),np.median(other[:,1]),np.median(other[:,2])]
min_ = [np.min(other[:,0]),np.min(other[:,1]),np.min(other[:,2])]
max_ = [np.max(other[:,0]),np.max(other[:,1]),np.max(other[:,2])]
std_ = [np.std(other[:,0]),np.std(other[:,1]),np.std(other[:,2])]

print(['mean=',avgs,'medians=',meds,'max=',max,'min=',min,'std=',std])
print(['mean=',avgs_,'medians=',meds_,'max=',max_,'min=',min_,'std=',std_])'''











