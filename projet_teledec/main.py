
### IMPORTS ###
import time
from pyrasta.raster import Raster
import gdal
from fototex.foto import Foto, FotoSector, FotoBatch
from fototex.foto_tools import degrees_to_cardinal
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functions import clustering, elbow, filtering, cluster_discrimination, histo_cluster, stats_clusters
import matplotlib.patches as mpatches
import h5py

#jp2_to_tif()

#### ALGORITHM FOTOTEX ####

# foto = Foto("/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/R10m/T54SUE_20210221T012659_B03_10m.jp2", band=None, method="block",
#          in_memory=True)
# foto.run(window_size=19,keep_dc_component=False)
# # # # #
# # # #
# foto.save_rgb()
# foto.save_data()
test= h5py.File('/home/gueneau/Documents/S2A_MSIL1C_20210321T163951_N0209_R126_T16SBC_20210321T203030.SAFE/GRANULE/L1C_T16SBC_A030010_20210321T164459/IMG_DATA/FOTO_method=block_wsize=19_dc=False_image=T16SBC_20210321T163951_B03_foto_data.h5')
r_spectra = test['r-spectra']



#### READING THE OUTPUT ###

img_path = '/home/gueneau/Images/T16/FOTO_method=block_wsize=19_dc=False_image=T16SBC_20210321T163951_B03_rgb.tif'
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





####  FILTERING ####



ker = (19,19)
img = filtering(ker,'Gaussian',img)




#### K-MEANS/CLUSTERING


'''
connectivity = kneighbors_graph(pixels,7)
K = 5                  ## nb of clusters'''
#
pixels = img.reshape((-1,3))
pixels = np.float32(pixels)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 70, 0.1)



#### ELBOW METHOD ###
# sil,db, delta_r = elbow(pixels,criteria)
# K = max(np.nanargmax(sil),np.nanargmin(db),np.nanargmin(delta_r)) + 1

K=3


labels, centers = clustering(img,'k_means',K)
#ward = clustering(gauss_17,'hierarchical',6)
#ward_labels = np.reshape(ward.labels_, (gauss_17.shape[0], gauss_17.shape[1]))





labels = labels.flatten()
#ward_labels = ward_labels.ravel()
segmented_image = centers[labels]
segmented_image = segmented_image.reshape(img.shape)




#### CLUSTER ELIMINATION/DISCRIMINATION ####

# masked_image, patches = cluster_discrimination(K,labels,img)
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

#### R-Spectra Analysis ###

ind_0 = np.where(labels==0)
ind_1 = np.where(labels==1)


n_labels = np.transpose([labels]*len(r_spectra[0]))
r_spectra0 = r_spectra[n_labels == 0].reshape(len(ind_0[0]),len(r_spectra[0]))
r_spectra1 = r_spectra[n_labels == 1].reshape(len(ind_1[0]),len(r_spectra[0]))
r_spectra_meanC0 = np.mean(r_spectra0,axis = 0)
r_spectra_meanC1 = np.mean(r_spectra1,axis = 0)
if K==3:
    ind_2= np.where(labels==2)
    r_spectra2 = r_spectra[n_labels == 2].reshape(len(ind_2[0]),len(r_spectra[0]))
    r_spectra_meanC2 = np.mean(r_spectra2,axis=0)


probas_spatiales = r_spectra[:,0]/max(r_spectra[:,0])



# ###### Contributions ######
# r_spectra_first_C0 = r_spectra0[:,0]
# r_spectra_first_C1 = r_spectra1[:,0]
# r_spectra_second_C0 = r_spectra0[:,1]
# r_spectra_second_C1= r_spectra1[:,1]
# r_spectra_third_C0  = r_spectra0[:,2]
# r_spectra_third_C1  = r_spectra1[:,2]
# r_spectra_fourth_C0 = r_spectra0[:,3]
# r_spectra_fourth_C1 = r_spectra1[:,3]
# r_spectra_fifth_C0 = r_spectra0[:,4]
# r_spectra_fifth_C1 = r_spectra1[:,4]
# r_spectra_sixth_C0 = r_spectra0[:,5]
# r_spectra_sixth_C1 = r_spectra1[:,5]
#
# ## Cluster 0
# plt.figure()
# plt.plot(r_spectra_sixth_C0)
# plt.xlabel('N° Window')
# plt.ylabel('Contribution')
# plt.title('Contributions for sixth r-spectra in non-urban cluster')
# plt.axhline(np.mean(r_spectra_sixth_C0),color='r',label='Mean')
# plt.axhline(np.median(r_spectra_sixth_C0),color='g',label='Median')
# plt.axis([0,len(r_spectra_sixth_C0),0,np.max(r_spectra_sixth_C1)])
# plt.legend()
# ## Cluster 1
# plt.figure()
# plt.plot(r_spectra_sixth_C1)
# plt.xlabel('N° Window')
# plt.ylabel('Contribution')
# plt.title('Contributions for sixth r-spectra in urban cluster')
# plt.axhline(np.mean(r_spectra_sixth_C1),color='r',label='Mean')
# plt.axhline(np.median(r_spectra_sixth_C1),color='g',label='Median')
# plt.axis([0,len(r_spectra_sixth_C1),0,np.max(r_spectra_sixth_C1)])
# plt.legend()


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






### Spectral Indices  ###

ndvi_raster = Raster('/home/gueneau/Documents/S2A_MSIL1C_20210321T163951_N0209_R126_T16SBC_20210321T203030.SAFE/GRANULE/L1C_T16SBC_A030010_20210321T164459/IMG_DATA/ndvi_t16.tif')
ndvi = ndvi_raster.read_array()
# ndvi = filtering(ker,'Gaussian',ndvi)
# # cv.imwrite('/home/gueneau/Documents/T37_NDVI_filtered.tif',ndvi)




ndwi_raster = Raster('/home/gueneau/Documents/S2A_MSIL1C_20210321T163951_N0209_R126_T16SBC_20210321T203030.SAFE/GRANULE/L1C_T16SBC_A030010_20210321T164459/IMG_DATA/ndwi_t16.tif')
ndwi = ndwi_raster.read_array()
# ndbi_raster = Raster('/home/gueneau/Documents/Indices/NDBI_T16_superimpose_bco.tif')
# ndbi = ndbi_raster.read_array()
mean, median , std = stats_clusters(dict,K,'red')
ib_raster = Raster('/home/gueneau/Documents/S2A_MSIL1C_20210321T163951_N0209_R126_T16SBC_20210321T203030.SAFE/GRANULE/L1C_T16SBC_A030010_20210321T164459/IMG_DATA/ib_t16.tif')
ib = ib_raster.read_array()

if K==4:
    ind_2 = np.where(labels==2)
    ind_3 = np.where(labels==3)

ndvi = ndvi.flatten()
ndwi = ndwi.flatten()
# ndbi = ndbi.flatten()
ib = ib.flatten()
# plt.figure()
# plt.hist(ndvi,bins='auto',label='NDVI')
# plt.hist(ndwi,bins='auto',label='NDWI')
# plt.hist(ndbi,bins='auto',label='NDBI')
# plt.legend()
#

ndvi_0 = ndvi[ind_0]
ndwi_0 = ndwi[ind_0]
ndvi_1 = ndvi[ind_1]
ndwi_1 = ndwi[ind_1]
# ndbi_0 = ndbi[ind_0]
# ndbi_1 = ndbi[ind_1]
ib_0 = ib[ind_0]
ib_1 = ib[ind_1]


if K==3:
    ndvi_2 = ndvi[ind_2]
    ndwi_2 = ndwi[ind_2]
    # ndbi_2 = ndbi[ind_2]
    ib_2 = ib[ind_2]
if K==4:
    ndvi_2 = ndvi[ind_2]
    ndwi_2 = ndwi[ind_2]
    # ndbi_2 = ndbi[ind_2]
    ndvi_3= ndvi[ind_3]
    ndwi_3= ndwi[ind_3]
    # ndbi_3 = ndbi[ind_3]




# plt.figure()
# cloud1 = plt.scatter(ndwi_0,ndvi_0,color='green')
# cloud2 = plt.scatter(ndwi_1,ndvi_1,color='blue')
# cloud3 = plt.scatter(ndwi_2,ndvi_2,color='orange')
# # # could4 = plt.scatter(ndwi_3,ndvi_3)
# plt.legend([cloud1,cloud2,cloud3],['Vegetation','Other','Built-up'])
# plt.xticks(np.arange(-1,1,0.1))
# plt.yticks(np.arange(-1,1,0.1))
# plt.xlabel('NDWI')
# plt.ylabel('NDVI')
# # plt.show()
#
# plt.figure()


# cloud1 = plt.scatter(ndvi_0,ndbi_0,linewidths=0.8,color='green')
# cloud2 = plt.scatter(ndvi_1,ndbi_1,linewidths=0.8,color='blue')
# cloud3 = plt.scatter(ndvi_2,ndbi_2,linewidths=0.8,color='orange')
# # cloud4 = plt.scatter(ndvi_3,ndbi_3)
# plt.legend([cloud1,cloud2,cloud3],['Vegetation','Other','Built-up'])
# plt.xticks(np.arange(-1,1,0.1))
# plt.yticks(np.arange(-1,1,0.1))
# plt.xlabel('NDVI')
# plt.ylabel('NDBI')
# plt.show()

ndvi_neg = np.where(ndvi<0)
ndwi_neg = np.where(ndwi<0)
ndvi[ndvi_neg] = 0
ndwi[ndwi_neg] = 0

probas_spectral = (1-ndwi)*(1-ndvi)
# probas_sup = np.where(probas_spectral>1)
# probas_spectral[probas_sup] = 1
probas = probas_spectral + probas_spatiales
probas = probas_spectral.reshape(577,577)

# # plt.figure()
# sample = np.random.choice([False,True], len(ndvi.flatten()), p=[0.95,0.05])
# ndvi_sample = ndvi[sample]
# ndwi_sample = ndwi[sample]
# ndbi_sample = ndbi[sample]
#



urban = np.argmax(median)
probas = np.zeros_like(labels)
probas = probas.astype(dtype='float32')
ind_urban_1 = np.where( (labels==urban) )
ind_urban_2 = np.where((( (labels==urban) & (abs(ndvi) <0.25)  )))
ind_urban_3 = np.where((( (labels==urban) & (abs(ndvi) <0.25) & (abs(ndwi) <0.25) )))
ind_urban_4 = np.where((( (labels==urban) & (abs(ndvi) <0.25) & (abs(ndwi) <0.25) & (ndbi > -0.2) )))
ind_urban_5  = np.where((labels != urban) & (abs(ndvi) < 0.25))

ind_vege = np.where(ndvi>0.3)
ind_water = np.where(ndwi>0.3)

for indx, indy in zip(ind_urban_5[0],ind_urban_5[1]):
        if foto[0][indx][indy]>5:
           img[indx][indy] = [1,0,0]
           probas[indx][indy] = 0.5
        else:
           img[indx][indy] = [0,0,0]
           probas[indx][indy] = 0


for indx,indy in zip(ind_urban_1[0],ind_urban_1[1]):
        img[indx][indy] = [0,1,0]
        probas[indx][indy] = 0.6

for indx,indy in zip(ind_urban_2[0],ind_urban_2[1]):
        probas[indx][indy] = 0.75

for indx,indy in zip(ind_urban_3[0],ind_urban_3[1]):
        img[indx][indy] = [0.5,0,1]
        probas[indx][indy] = 0.9

for indx,indy in zip(ind_urban_4[0],ind_urban_4[1]):
        img[indx][indy] = [1,1,1]
        probas[indx][indy] = 1

for indx,indy in zip(ind_vege[0],ind_vege[1]):
        probas[indx][indy] = 0
        img[indx][indy] = [0, 0, 0]

for indx,indy in zip(ind_water[0],ind_water[1]):
        probas[indx][indy] = 0
        img[indx][indy] = [0, 0, 0]

colors = [[1,1,1],[0.5,0,1],[0,1,0],[1,0,0],[0,0,0]]
values = [1,0.9,0.6,0.5,0]
patches = [ mpatches.Patch(color = colors[i], label= "Proba = {c}".format(c=values[i])) for i in range(len(values))]



plt.imshow(img)
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.show()
print(0)




# mean_red = np.mean(dict[1][0])


# accuracy =
# precision =
# recall =
# f1_score = 2*precision*recall/(precision+recall)












