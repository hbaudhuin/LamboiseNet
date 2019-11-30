import numpy as np
from PIL import Image
#import tensorflow as tf
import imgaug.augmenters as iaa

def rotate4(RGBarrays_list):
    # take a list, return a list, this design point is to be discussed, we could use numpy arrays instead, optimize memory usage
    all = []
    for RGBarrays in RGBarrays_list:
        ret = [RGBarrays.copy(), RGBarrays.copy(), RGBarrays.copy(), RGBarrays.copy()]
        ret[1] = np.rot90(ret[0])
        ret[2] = np.rot90(ret[1])
        ret[3] = np.rot90(ret[2])
        all.append(ret[0])
        all.append(ret[1])
        all.append(ret[2])
        all.append(ret[3])
    return all

def horizontalFlip(RGBarrays_list) :

    flipped = np.zeros(RGBarrays_list.shape)
    for index ,img in enumerate(RGBarrays_list ):
        i =0
        for array in img[:][:] :
            flipped[index][:][:][i] = np.flipud(array)
            i+=1
    """  print(flipped.shape)
    img1= np.zeros((650, 650, 3))
    img1[:,:,0] = flipped[0][0]
    img1[:, :, 1] = flipped[0][1]
    img1[:, :, 2] = flipped[0][2]
    print(img1.shape)
    img = Image.fromarray((img1 * 255).astype(np.uint8))
    img.save('horizontalFlip.jpeg')
"""

    return flipped


def verticalFlip(RGBarrays_list):
    flipped = np.zeros(RGBarrays_list.shape)
    for index, img in enumerate(RGBarrays_list):
        i = 0
        for array in img[:][:]:
            flipped[index][:][:][i] = np.fliplr(array)
            i += 1
    return flipped


def sharpen(image, tuple):
    seq = iaa.Sequential(
        [iaa.Affine(rotate=tuple)]
    )
    images_aug = seq(images=image)
    return images_aug

def gaussianBlur(image, strength) :
    gauss = iaa.Sequential([iaa.GaussianBlur(sigma = strength)])
    res = gauss(images=image)
    return res

def hueAndSaturation(image, range):
    modifier = iaa.AddToHueAndSaturation(range, per_channel=True)
    print(image)
    res = modifier(images=image.astype(float))
    return res

def rgb_modifications(images, alpha, beta):
    # rand1 = np.randint(-255, 255)

    res = images.copy()
    i =0
    for image in images :
        res[i][:][:] = image[:][:] * alpha + beta
        #print("result")
       # print(res[i])
       # print("modified")
       # print(image)
        #print("TEST")
       # print(image[:][:]*125 + 4000)
        i+=1
    return res





