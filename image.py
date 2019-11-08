import numpy as np
import tifffile as tif
from PIL import Image

image_num = "1180"

image = tif.imread("Paris_"+image_num+"/RGB-PanSharpen_AOI_3_Paris_img"+image_num+".tif")
sh = image.shape
print(sh)

arrs = [None, None, None]
for i in range(3):
    #arr = np.copy(image[i][-4000:, -4000:])
    arr = np.copy(image[...,i])
    mx = arr.max()
    arr = arr / (1.0 * mx) * 250
    av = arr.mean()
    print(av)
    arr = arr * 150 / av
    arr[arr > 250] = 250
    #print(arr.shape)
    arrs[i] = arr


print("merging")
rgbArray = np.zeros((650,650,3), 'uint8')
rgbArray[..., 0] = arrs[0]
rgbArray[..., 1] = arrs[1]
rgbArray[..., 2] = arrs[2]

print("saving")
img = Image.fromarray(rgbArray)
img.save('Paris_'+image_num+'/out_'+image_num+'.png')
