import time


import json, re, itertools, os

import numpy
from pyrasta.raster import Raster
import geopandas
from osgeo import ogr

import gdal
from fototex.foto import Foto, FotoSector, FotoBatch
from fototex.foto_tools import degrees_to_cardinal
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
from functions import clustering, elbow, filtering, cluster_discrimination, histo_cluster, stats_clusters
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

#jp2_to_tif()

#### ALGORITHM FOTOTEX ####

#foto = Foto("/home/paul/PycharmProjects/projet_teledec/data/T54SUE_20210427T012651_B03.jp2", band=None, method="block",
          #in_memory=True, data_chunk_size=50000)
#foto.run(window_size=19,keep_dc_component=False)

#foto.save_rgb()


#### READING THE OUTPUT ###

img_path = '/media/gueneau/D0F6-1CEA/imgs/FOTO_method=block_wsize=13_dc=False_image=T37RGL_20210521T074611_B03_rgb.tif'
dataset = gdal.Open(img_path)

### Fetching the channels ###
band1 = dataset.GetRasterBand(1)

band2 = dataset.GetRasterBand(2)
band3 = dataset.GetRasterBand(3)
b1 = band1.ReadAsArray()
b2 = band2.ReadAsArray()
b3 = band3.ReadAsArray()


### Display the arrays
img = np.dstack((b1,b2,b3))



### MASKS ###

### A part conceived originally to highlight some parts of the image by masking the others ####


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




#### K-MEANS/CLUSTERING


'''
connectivity = kneighbors_graph(pixels,7)
K = 5                  ## nb of clusters'''

pixels = img.reshape((-1,3))
pixels = np.float32(pixels)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 70, 0.1)



#### ELBOW METHOD ###
sil,db, delta_r = elbow(pixels,criteria)
K = max(np.nanargmax(sil),np.nanargmin(db),np.nanargmin(delta_r)) + 1




labels, centers = clustering(img,'k_means',K)
#ward = clustering(gauss_17,'hierarchical',6)
#ward_labels = np.reshape(ward.labels_, (gauss_17.shape[0], gauss_17.shape[1]))

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

# cv.imwrite("/media/gueneau/D0F6-1CEA/imgs/Clusters.tif",segmented_image)


#### CLUSTER ELIMINATION/DISCRIMINATION ####

#masked_image, patches = cluster_discrimination(K,labels,img)





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
foto_raster = Raster(img_path)


foto = foto_raster.read_array()
dict = histo_cluster(labels,foto,K)

# fig, ax = plt.subplots(1, K)
# for i in range(K):
#     ax[i].hist(dict[i][0], bins='auto', color="red")
#     ax[i].hist(dict[i][1], bins='auto', color="blue")
#     ax[i].hist(dict[i][2], bins='auto', color="green")
#     ax[i].set_title('Cluster ='+str(i))
#     ax[i].axis([-50, 50, 0, 6000 ])
# plt.show()




### Print RGB stats for each cluster ###

ndvi_raster = Raster('/home/gueneau/Documents/NDVI_T37_superimpose_bco.tif')
ndvi = ndvi_raster.read_array()
# ndvi = filtering(ker,'Gaussian',ndvi)
# # cv.imwrite('/home/gueneau/Documents/T37_NDVI_filtered.tif',ndvi)


ndwi_raster = Raster('/home/gueneau/Documents/NDWI_T37_superimpose_bco.tif')
ndwi = ndwi_raster.read_array()
ib_raster = Raster('/home/gueneau/Documents/IB_T37_superimpose_bco.tif')
ib = ib_raster.read_array()
# mean = stats_clusters(dict,K,'red')
ind_0 = np.where(labels==0)
ind_1 = np.where(labels==1)
if K==3:
    ind_2= np.where(labels==2)

probas = np.zeros_like(labels)
probas = probas.astype(dtype='float32')
ind_urban_1 = np.where( (labels==0) )
ind_urban_2 = np.where((( (labels==0) & (ndvi <0.25)  )))
ind_urban_3 = np.where((( (labels==0) & (ndvi <0.25) & (ndwi <0.25) )))
ind_urban_4 = np.where((( (labels==0) & (ndvi <0.25) & (ndwi <0.25) & (ib > 3000) )))


ind_vege = np.where(ndvi>0.3)
ind_water = np.where(ndwi>0.3)

for indx,indy in zip(ind_urban_1[0],ind_urban_1[1]):
        probas[indx][indy] = 0.6

for indx,indy in zip(ind_urban_2[0],ind_urban_2[1]):
        probas[indx][indy] = 0.75

for indx,indy in zip(ind_urban_3[0],ind_urban_3[1]):
        probas[indx][indy] = 0.9

for indx,indy in zip(ind_urban_4[0],ind_urban_4[1]):
        img[indx][indy] = [1,1,1]
        probas[indx][indy] = 1

for indx,indy in zip(ind_vege[0],ind_vege[1]):
        probas[indx][indy] = 0

for indx,indy in zip(ind_water[0],ind_water[1]):
        probas[indx][indy] = 0



plt.imshow(img)
plt.show()
print(0)




# mean_red = np.mean(dict[1][0])


# accuracy =
# precision =
# recall =
# f1_score = 2*precision*recall/(precision+recall)












