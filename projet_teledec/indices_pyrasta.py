from pyrasta.raster import Raster
import numpy as np

blue = Raster('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/T45TUK_20210526T051651_B02.jp2')
green = Raster('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/T45TUK_20210526T051651_B03.jp2')
red = Raster('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/T45TUK_20210526T051651_B04.jp2')
nir = Raster('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/T45TUK_20210526T051651_B08.jp2')

ndwi = (green - nir)/(nir + green)
ndvi = (nir-red)/(nir+red)
ndvi_window19 = ndvi.windowing(np.mean,window_size=19,method='block')
ndwi_window19 = ndwi.windowing(np.mean,window_size=19,method='block')
ndvi_window19.to_file('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/ndvi_T45.tif')
ndwi_window19.to_file('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/ndwi_T45.tif')
def ib(bands):
    r,g,b,nir = bands
    return (r**2+g**2+b**2+nir**2)**(1/2)


ib_19 = Raster.raster_calculation([blue,green,red,nir],ib)
ib_19 = ib_19.windowing(np.mean,window_size=19,method='block')
ib_19.to_file('/media/gueneau/D0F6-1CEA/S2A_MSIL1C_20210526T051651_N0300_R062_T45TUK_20210526T073136.SAFE/GRANULE/L1C_T45TUK_A030947_20210526T051651/IMG_DATA/ib_T45.tif')