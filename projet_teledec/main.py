
### IMPORTS ###
import time

from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from pyrasta.raster import Raster
import gdal
from fototex.foto import Foto, FotoSector, FotoBatch
from fototex.foto_tools import degrees_to_cardinal
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functions import clustering, elbow, filtering, cluster_discrimination, histo_cluster, stats_clusters, filter_nan_gaussian_conserving
import matplotlib.patches as mpatches
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
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
# test= h5py.File('/home/gueneau/Documents/S2A_MSIL1C_20210321T163951_N0209_R126_T16SBC_20210321T203030.SAFE/GRANULE/L1C_T16SBC_A030010_20210321T164459/IMG_DATA/FOTO_method=block_wsize=19_dc=False_image=T16SBC_20210321T163951_B03_foto_data.h5')
# r_spectra = test['r-spectra']
#
path_img_sentinel2 = '/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210426T083601_N0300_R064_T34KBG_20210426T104005.SAFE/GRANULE/L1C_T34KBG_A030520_20210426T090203/IMG_DATA/T34KBG_20210426T083601_B03.jp2'
img_sentinel2 = Raster(path_img_sentinel2)



#### READING THE OUTPUT ###
start = time.time()
img_path = '/home/gueneau/Documents/FOTO_Clustering_T16/FOTO_method=block_wsize=19_dc=False_image=masked_sentinel2_rgb.tif'
dataset = gdal.Open(img_path)

foto_raster = Raster(img_path)
img = foto_raster.read_array()
img = np.moveaxis(img,0,-1)

width,height = img.shape[0],img.shape[1]
### Fetching the channels ###
# band1 = dataset.GetRasterBand(1)
#
# band2 = dataset.GetRasterBand(2)
# band3 = dataset.GetRasterBand(3)
# b1 = band1.ReadAsArray()
# b2 = band2.ReadAsArray()
# b3 = band3.ReadAsArray()


### Display the arrays
# img = np.dstack((b1,b2,b3))





###  FILTERING #### #### K-MEANS/CLUSTERING
#
# ### Itération 1 -> pas de NaN values, sortie FOTO entière
#
# ker = (9,9)
#
# img  = filtering(ker,'Gaussian',img)
# img = img.reshape(-1,3)
# K=2
# labels, centers = clustering(img,'k_means',K)
# labels = labels.flatten()
# masked_image, patches = cluster_discrimination(K,labels,img)


### Itérations suivantes -> NaN values, sortie FOTO masquée
img[img == -999] = np.nan  #
img = filter_nan_gaussian_conserving(img,1)
# img = img.reshape(-1,3)
kernel = Gaussian2DKernel(x_stddev=2,y_stddev=2)
# img = convolve(img,kernel,nan_treatment='interpolate')
# img_interp = interpolate_replace_nans(img, kernel)

# img = img.reshape(577,577,3)
#img_filter  =img_filter.reshape(577,577,3)
K=2
img_non_nan = img.reshape(-1,3)
img_non_nan = img_non_nan[~np.isnan(img_non_nan[:,0]),:]
new_labels = np.full((width,height),np.nan)
img_non_nan = img_non_nan.astype('float32')
labels, centers = clustering(img_non_nan,'k_means',K)


img = img.reshape(-1,3)
boolean = ~np.isnan(img[:,0])
boolean = boolean.reshape((width,height))
new_labels[boolean] = labels.flatten()
new_labels = new_labels.reshape(width,height)
new_labels = new_labels.flatten()
masked_image, patches = cluster_discrimination(K,new_labels,img)


### LABELLING ###
# new_labels[boolean] = labels.flatten()
new_labels = new_labels.reshape(width,height)
new_labels[np.isnan(new_labels)] = 0
label_raster = Raster.from_array(new_labels, foto_raster.crs, foto_raster.bounds)
label_raster = label_raster.resample(19)
label_raster.to_file("/home/gueneau/Documents/label_raster_t34kbg_3_conserv.tif")
# ward = clustering(img_non_nan,'hierarchical',K)
#ward_labels = np.reshape(ward.labels_, (width, height))

#ward_labels = ward_labels.ravel()


#
# connectivity = kneighbors_graph(pixels,7)

# pixels = img.reshape((-1,3))
# pixels = np.float32(pixels)
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 70, 0.1)
#
#

#### ELBOW METHOD ###
# sil,db, delta_r = elbow(pixels,criteria)
# K = max(np.nanargmax(sil),np.nanargmin(db),np.nanargmin(delta_r)) + 1
# end = time.time()
# print(end-start)








#### CLUSTER ELIMINATION/DISCRIMINATION ####

# masked_image, patches = cluster_discrimination(K,labels,img)
# #







start_2 = time.time()
labels = labels.reshape(577,577)
ind_0 = np.where(labels==0)
ind_1 = np.where(labels==1)
ind_2 = np.where(labels == 2)
# labels[ind_0] = 1
# labels[ind_1] = 0


# label_raster = Raster.from_array(labels, foto_raster.crs, foto_raster.bounds)
# label_raster = Raster.from_array(new_labels, foto_raster.crs, foto_raster.bounds)
# label_raster = label_raster.resample(19)
# label_raster.to_file("/home/gueneau/Documents/label_raster_t34kbg.tif")


#### R-Spectra Analysis ###
# n_labels = np.transpose([labels]*len(r_spectra[0]))
# r_spectra0 = r_spectra[n_labels == 0].reshape(len(ind_0[0]),len(r_spectra[0]))
# r_spectra1 = r_spectra[n_labels == 1].reshape(len(ind_1[0]),len(r_spectra[0]))
# r_spectra_meanC0 = np.mean(r_spectra0,axis = 0)
# r_spectra_meanC1 = np.mean(r_spectra1,axis = 0)
# if K==3:
#     ind_2= np.where(labels==2)
#     r_spectra2 = r_spectra[n_labels == 2].reshape(len(ind_2[0]),len(r_spectra[0]))
#     r_spectra_meanC2 = np.mean(r_spectra2,axis=0)
# if K==4:
#     ind_2 = np.where(labels == 2)
#     ind_3 = np.where(labels == 3)
#     r_spectra2 = r_spectra[n_labels == 2].reshape(len(ind_2[0]), len(r_spectra[0]))
#     r_spectra3 = r_spectra[n_labels == 3].reshape(len(ind_3[0]), len(r_spectra[0]))
#     r_spectra_meanC2 = np.mean(r_spectra2, axis=0)
#     r_spectra_meanC3 = np.mean(r_spectra3, axis=0)



img = img.reshape(width,height,3)
no_data = -999
# img[new_labels==0] = no_data
img = np.moveaxis(img,-1,0)
img_raster = Raster.from_array(img,foto_raster.crs,foto_raster.bounds,no_data=no_data)
new_img = img_raster.read_array()
#new_img = np.moveaxis(new_img,0,-1)
clip_sentinel2 = img_sentinel2.clip(bounds=img_raster.bounds)
clip_sentinel2.to_file("/home/gueneau/Documents/clip_t34kbg_sentinel2.tif")
#foto = Foto("/home/gueneau/Documents/output_FOTO_with_nan_values.tif", band=None, method="block",
       #   in_memory=True)
#foto.run(window_size=19,keep_dc_component=False)
#foto.save_rgb()


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

labels = labels.reshape(width,height)
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

ndvi_raster = Raster('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/ndvi_T45.tif')
ndvi = ndvi_raster.read_array()
# ndvi = filtering(ker,'Gaussian',ndvi)
# # cv.imwrite('/home/gueneau/Documents/T37_NDVI_filtered.tif',ndvi)




ndwi_raster = Raster('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/ndwi_T45.tif')
ndwi = ndwi_raster.read_array()
# ndbi_raster = Raster('/home/gueneau/Documents/Indices/NDBI_T16_superimpose_bco.tif')
# ndbi = ndbi_raster.read_array()
mean, median , std = stats_clusters(dict,K,'red')
ib_raster = Raster('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/ib_T45.tif')
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
end= time.time()
print(end-start_2)
ndvi_neg = np.where(ndvi<0)
ndwi_neg = np.where(ndwi<0)
ndvi[ndvi_neg] = 0
ndwi[ndwi_neg] = 0

probas_spectral = (1-ndwi)*(1-ndvi)
# probas_sup = np.where(probas_spectral>1)
# probas_spectral[probas_sup] = 1
probas = probas_spectral
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
ind_vege = np.where(ndvi>0.3)
ind_water = np.where(ndwi>0.3)
















