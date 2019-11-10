import numpy as np
import tifffile as tif
from PIL import Image
import imageio

image_num = "1180"

# Loading a tif file
#image = tif.imread("Paris_"+image_num+"/RGB-PanSharpen_AOI_3_Paris_img"+image_num+".tif")

# Loading a png file
image = imageio.imread("Paris_"+image_num+"/after.png")

sh = image.shape
print(sh)

# R G B, with sometimes a 4th Alpha channel on PNG
arrs = [None, None, None]
for i in range(3):
    #arr = np.copy(image[i][-4000:, -4000:])
    arr = np.copy(image[...,i])

    # Normalization between 0 and 250
    mx = arr.max()
    arr = arr / (1.0 * mx) * 250

    # Normalization using the average value of the channel
    av = arr.mean()
    print(av)
    arr = arr * 100 / av

    # Limiting the maximal value
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

print("done")
