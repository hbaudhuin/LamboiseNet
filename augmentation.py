import imgaug.augmenters as iaa
import numpy as np


def applyAugmentation(images):
    """
    Aplly a random number of augmentation to the images.
    OneOf = exactly one
    someOf = a few
    Sometimes(prob, if, else)
    WithChannels(channels=None,

    :param images:
    :return:
    """

    #rotation
    rotate_bounds = (0, 20)
    #cropping
    crop_bounds = (0, 0.3)
    #shearing
    tuple = (0.15, 20.0)
    #scale
    scalex_bound = (0.3, 1.5)
    scaley_bound = (0.3, 1.7)
    #blur
    sigma = (np.random.uniform(0.2, 2.5), np.random.uniform(0.2, 2.5))
    #contrast
    gamma = (0.75,1.1)
    #noise
    scale = (0.6, 4)
    #hue
    range = (0, 20)

    shear_value = np.random.randint(tuple[0], tuple[1])
    rotate_value = np.random.randint(rotate_bounds[0], rotate_bounds[1])
    crop_value = np.random.uniform(crop_bounds[0], crop_bounds[1])
    scalex = np.random.uniform(scalex_bound[0], scalex_bound[1])
    scaley = np.random.uniform(scaley_bound[0], scaley_bound[1])
    gamma = np.random.uniform(gamma[0], gamma[1])


    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    sequence_on_all_images = iaa.Sequential(
          [#sometimes(iaa.Affine(shear=(shear_value, shear_value))),
           iaa.Sometimes(0.2,iaa.Affine(rotate=(rotate_value, rotate_value)))
           #iaa.Fliplr(1),
           #iaa.Flipud(1),
           #iaa.Crop(percent=(crop_value, crop_value)),
           #iaa.Affine(scale={"x": (scalex, scalex), "y": (scaley, scaley)})
           ]
    )
    output = convertBackToUint(sequence_on_all_images(images = images))

    sequence_not_on_mask = iaa.Sequential([
        iaa.Sometimes(0.3,iaa.GaussianBlur(sigma =sigma))
        #iaa.GammaContrast(gamma = gamma),
        #iaa.AddToHueAndSaturation(range, per_channel=True),
        #iaa.AdditiveGaussianNoise(loc=0, scale=scale, per_channel=0.5)
    ])
    output[0:2] = sequence_not_on_mask(images = output[0:2] )
    return convertBackToUint(output)


def verticalFlip(RGBarrays_list) :
    """
    Perform a vertical flip on the three images given in input
    :param RGBarrays_list: The three input images in one matrix 650x650x(3+3+3)
    :return: The three input images in one matrix 650x650x(3+3+3) flipped on a vertical axis
    """
    flipped = np.zeros(RGBarrays_list.shape)
    for index, img in enumerate(RGBarrays_list):

        for index2 in [0,1,2]:
            array = img[:,:,index2]
            flipped[index,:,:,index2] = np.fliplr(array)

    return convertBackToUint(flipped)


def horizontalFlip(RGBarrays_list):
    """
    Perform a horizontal flip on the three images given as input
    :param RGBarrays_list: The three input images in one matrix 650x650x(3+3+3)
    :return: The three input images in one matrix 650x650x(3+3+3) flipped on a horizontal axis
    """

    flipped = np.zeros(RGBarrays_list.shape)
    for index, img in enumerate(RGBarrays_list):

        for index2 in [0,1,2]:
            array = img[:,:,index2]
            flipped[index,:,:,index2] = np.flip(array,1)
    return convertBackToUint(flipped)


def shear(image, tuple):
    """
    Shear the images
    :param image: RGB matrix of 3 input images
    :param tuple: tuple with min and max bound of the shearing
    :return: RGB matrix of the sheared images
    """
    shear_value =np.random.randint(tuple[0], tuple[1])
    seq = iaa.Sequential([iaa.Affine(shear=(shear_value, shear_value))])
    images_aug = seq(images=image)
    return convertBackToUint(images_aug)

def gaussianBlur(image, strength) :
    """
    Blur an image with a Gautian Blur
    :param image: the image(s) to be blurred. Can be multiple, blurring is independent of the number of images
    :param strength: tuple contain min valu and max value of the sigma. True value will be random between values.
    :return: RGB matrix of the input with blur
    """
    res = image.copy()
    gauss = iaa.Sequential([iaa.GaussianBlur(sigma = strength)])
    images_to_modified = image[0:2]
    res[0:2] = convertBackToUint(gauss(images=images_to_modified))
    return res

def hueAndSaturation(image, range):
    """
    Change hue and saturation by value in range given
    :param image: RGB matrix of the two images. (uselles augmentation on a mask)
    :param range: range of hue and saturation modification (-255, 255)
    :return: RGB matrix
    """
    res = image.copy()
    modifier = iaa.AddToHueAndSaturation(range, per_channel=True)
    res[0:2] = modifier(images=convertBackToUint(image[0:2]))
    return res


def rotate(images, bounds):
    """
    Rotate the image by a random angle between the min and max value of the bounds given
    :param images: RGB matrix of all images
    :param bounds: min, max value of the rotating angle
    :return:  RGB matrixes of all images rotated
    """
    rotate_value = np.random.randint(bounds[0], bounds[1])
    rotate = iaa.Affine(rotate=(rotate_value, rotate_value))
    rotated_image = rotate(images = images)
    return convertBackToUint(rotated_image)

def gaussianNoise(images, bounds):
    """
    adding gaussian noise to images. Different noise for each image.
    :param images: RGB matrix of two input images. Nois on mask is useless
    :param bounds: min and max of the gaussian noise
    :return: RGB matrix with gaussian Noise
    """
    noisy_images = images.copy()
    gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=bounds, per_channel=0.5)
    noisy_images[0:2] = gaussian_noise(images = convertBackToUint(images[0:2]))
    return noisy_images


def crop(images, bounds):
    """

    :param images: RGB matrixes of input images including mask
    :param bounds: min and max of percetnage of cropping
    :return:
    """
    crop_value = np.random.uniform(bounds[0], bounds[1])
    crop = iaa.Crop(percent=(crop_value, crop_value))
    cropped_images = crop(images = images)
    return convertBackToUint(cropped_images)


def contrast(images, gamma):
    """
    Modifie the contrast of an image by a gamma distribution.
    :param images: RGB matrixes of two images. contrast on mask is useless
    :param gamma: value of gamma parameter (constraint to 0.8 and 1.05) If too high or low, the image is either black or overly saturated
    :return: RBG matrix with contrast changed
    """
    contrasted_images = images.copy()
    contraster = iaa.GammaContrast(gamma=gamma)
    contrasted_images[0:2] = contraster(images = images[0:2])
    return convertBackToUint(contrasted_images)


def scale(images, boundx, boundy):
    """
    Scale the image on the x axis by a random value between boundx and on the y axis by a random value between boundy
    :param images: RGB matrixes of all images including mask
    :param boundx: min and max of the distance moved in x
    :param boundy:  min and max of the distance moved in y
    :return: RGB matrixes scaled
    """
    scalex = np.random.uniform(boundx[0], boundx[1])
    scaley = np.random.uniform(boundy[0], boundy[1])
    scaler = iaa.Affine(scale ={"x": (scalex, scalex), "y": (scaley, scaley)})
    scaled_images = scaler(images = images)
    return convertBackToUint(scaled_images)



def convertBackToUint(matrix):
    matrix = matrix.astype(np.float64) / 255
    data = 255 * matrix
    img = data.astype(np.uint8)
    return img





