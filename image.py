import numpy as np
import tifffile as tif
from PIL import Image
import imageio


image_num = [276, 277, 278, 280,
             301, 302, 303, 304, 305,
             328, 332,
             356, 357, 358, 359, 360,
             385, 387, 389]

def main():

    '''
    for im_num in image_num:
        print(im_num)
        image = open_image("../AOI_3_Paris_Train/AOI_3_Paris_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_3_Paris_img"+str(im_num)+".tif")
        arrs = normalize(image)
        save_image(arrs, "DATA/patch1/out_"+str(im_num)+".png")
    '''

    image = open_image("DATA/patch1/patch1_after.png")
    arrs = normalize(image)
    save_image(arrs, "DATA/patch1/patch1_after_norm.png")


def open_image(filename):
    print("opening")

    # Loading a tif file
    if filename[-4:].lower() in [".tif", "tiff"]:
        return tif.imread(filename)
    # Loading a png file
    else:
        return imageio.imread(filename)


def normalize(image):
    print("normalizing")

    # R G B, with sometimes a 4th Alpha channel on PNG
    arrs = [None, None, None]
    for i in range(3):

        arr = np.copy(image[...,i])

        # Normalization between 0 and 255
        mx = arr.max()
        arr = arr / (1.0 * mx) * 255

        # Normalization using the average value of the channel
        #av = arr.mean()
        #print(av)
        #arr = arr * 100 / av

        # Remove outliers and normalize
        hist = np.sort(arr.flatten())
        lo = hist[int(0.025 * len(hist))]
        hi = hist[int(0.975 * len(hist))]
        arr = (arr-lo) / (hi-lo) * 255.0

        # Limiting the min and maximal value
        arr[arr < 0] = 0
        arr[arr > 255] = 255

        #print(arr.shape)
        arrs[i] = arr

    return arrs


def save_image(arrs, location):
    print("merging")
    rgbArray = np.zeros((arrs[0].shape[0], arrs[0].shape[1], 3), 'uint8')
    rgbArray[..., 0] = arrs[0]
    rgbArray[..., 1] = arrs[1]
    rgbArray[..., 2] = arrs[2]

    print("saving")
    img = Image.fromarray(rgbArray)
    img.save(location)


if __name__ == '__main__':
    import time
    t_start = time.time()

    main()
    print("\ndone\n")

    t_end = time.time()
    print("total time : " + str(int((t_end - t_start)*1000)/1000.0) + " sec")
