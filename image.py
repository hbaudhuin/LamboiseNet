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

    '''
    image = open_image("DATA/patch1/patch1_after.png")
    arrs = normalize(image)
    save_image(arrs, "DATA/patch1/patch1_after_norm.png")
    '''

    inputs, masks = load_dataset(["1180", "1180", "1180"])
    print(inputs.shape)
    print(masks.shape)

def open_image(filename):
    print("opening")

    # Loading a tif file
    if filename[-4:].lower() in [".tif", "tiff"]:
        return tif.imread(filename)
    # Loading a png file
    else:
        return imageio.imread(filename)


def image_to_arrs(image):
    arrs = [None, None, None]
    for i in range(3):
        arr = np.copy(image[..., i])
        arrs[i] = arr
    return arrs


def images_prepare(img_before, img_after, img_mask):
    i_b = img_before[..., [0, 1, 2]]
    i_a = img_after[..., [0, 1, 2]]
    i_m = rgb_to_grey(img_mask)
    i_join = np.zeros(shape=(6, i_b.shape[0], i_b.shape[1]))
    i_join[[0, 1, 2], ...] = np.transpose(i_b[...], axes=(2, 0, 1))
    i_join[[3, 4, 5], ...] = np.transpose(i_a[...], axes=(2, 0, 1))
    return i_join, i_m


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


def rgb_to_grey(mask):
    grey = np.zeros(shape=(mask.shape[0], mask.shape[1]))
    grey[...] = np.average(mask[:, ...])
    return grey


def load_dataset(img_nums):
    inputs = np.zeros(shape=(len(img_nums), 6, 650, 650)) # TODO un-hardcode
    masks = np.zeros(shape=(len(img_nums), 650, 650))
    for i, img_num in enumerate(img_nums):
        img_b = open_image("DATA/Paris_" + str(img_num) + "/before.png")
        img_a = open_image("DATA/Paris_" + str(img_num) + "/after.png")
        img_m = open_image("DATA/Paris_" + str(img_num) + "/mask.png")
        input, mask = images_prepare(img_b, img_a, img_m)
        inputs[i] = input
        masks[i] = mask
    return inputs, masks


if __name__ == '__main__':
    import time
    t_start = time.time()

    main()
    print("\ndone\n")

    t_end = time.time()
    print("total time : " + str(int((t_end - t_start)*1000)/1000.0) + " sec")
