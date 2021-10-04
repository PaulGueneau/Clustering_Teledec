from pyrasta.raster import Raster
import numpy as np

blue = Raster('/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/R10m/T54SUE_20210221T012659_B02_10m.jp2')
green = Raster('/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/R10m/T54SUE_20210221T012659_B03_10m.jp2')
red = Raster('/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/R10m/T54SUE_20210221T012659_B04_10m.jp2')
nir = Raster('/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/R10m/T54SUE_20210221T012659_B08_10m.jp2')

ndwi = (green - nir)/(nir + green)
ndvi = (nir-red)/(nir+red)
ndvi_window19 = ndvi.windowing(np.mean,window_size=19,method='block')
ndwi_window19 = ndwi.windowing(np.mean,window_size=19,method='block')
ndvi_window19.to_file('/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/ndvi_t54.tif')
ndwi_window19.to_file('/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/ndwi_t54.tif')
def ib(bands):
    r,g,b,nir = bands
    return (r**2+g**2+b**2+nir**2)**(1/2)


ib_19 = Raster.raster_calculation([blue,green,red,nir],ib)
ib_19 = ib_19.windowing(np.mean,window_size=19,method='block')
ib_19.to_file('/home/gueneau/Documents/S2B_MSIL2A_20210221T012659_N0214_R074_T54SUE_20210221T040051.SAFE/GRANULE/L2A_T54SUE_A020692_20210221T012653/IMG_DATA/R10m/ib_t54.tif')