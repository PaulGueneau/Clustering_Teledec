import gdal
import numpy as np
from pyrasta.raster import Raster


#def ndvi(arrays):
#    return (arrays[0] - arrays[1])/(arrays[0] + arrays[1])


#def ib(*channels):
#    return np.sqrt(np.sum([c**2 for c in channels]))


nir = Raster("/media/paul/D0F6-1CEA/S2A_MSIL1C_20210521T074611_N0300_R135_T37RGL_20210521T092215.SAFE/GRANULE/"
             "L1C_T37RGL_A030877_20210521T075213/IMG_DATA/T37RGL_20210521T074611_B08.jp2")
r = Raster("/media/paul/D0F6-1CEA/S2A_MSIL1C_20210521T074611_N0300_R135_T37RGL_20210521T092215.SAFE/GRANULE/"
           "L1C_T37RGL_A030877_20210521T075213/IMG_DATA/T37RGL_20210521T074611_B04.jp2")
b = Raster("/media/paul/D0F6-1CEA/S2A_MSIL1C_20210521T074611_N0300_R135_T37RGL_20210521T092215.SAFE/GRANULE/"
           "L1C_T37RGL_A030877_20210521T075213/IMG_DATA/T37RGL_20210521T074611_B02.jp2")
g = Raster("/media/paul/D0F6-1CEA/S2A_MSIL1C_20210521T074611_N0300_R135_T37RGL_20210521T092215.SAFE/GRANULE/"
           "L1C_T37RGL_A030877_20210521T075213/IMG_DATA/T37RGL_20210521T074611_B03.jp2")



#ndvi = (nir - r)/(nir + r)
ib = (b**2+g**2+nir**2+r**2)**0.5


#ndvi.to_file("/media/paul/D0F6-1CEA/NDVI/NDVI_T37.tif")
ib.to_file("/media/paul/D0F6-1CEA/NDVI/IB_T37.tif")
#ndvi_raster = Raster.raster_calculation(rasters=[nir, r], fhandle=ndvi,
#                                        output_type=gdal.GetDataTypeByName("float32"))
