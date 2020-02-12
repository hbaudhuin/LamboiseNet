import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmenters import Affine


def applyAugmentation(images):
    """
    Aply augmentations to the images. Augmentation are : vertical flipping, horzontal flipping, rotation, cropping,
    shearing, scaling, blurring, contrast adjusting, adding noise and hue and saturation change. They are appied with a
    certain probability.
    :param images: RGB matrixes of two images and the corresponding mask
    :return: RGB matrixes of two images and the corresponding mask with modification applied.
    """
    # HYPER-PARAMETERS
    # rotation
    rotate_bounds = (0, 10)
    # cropping
    crop_bounds = (0, 0.15)
    # shearing
    tuple = (0.15, 6.0)
    # scale
    scalex_bound = (1.0, 1.3)
    scaley_bound = (1.0, 1.3)
    # blur
    sigma = (np.random.uniform(0.2, 2.5), np.random.uniform(0.2, 2.5))
    # contrast
    gamma = (0.75, 1.1)
    # noise
    scale_noise = (10, 30)
    # hue
    hue_range = (-30, 50)

    output = images
    dice = np.random.randint(0,10) / 10.0
    if dice < 0.3 :
        output = shear(images, tuple)
    if dice < 0.3 :
        output = rotate(output, rotate_bounds)

    for i in range(0,5):
        dice = np.random.randint(0, 10) / 10.0
        if dice < 0.4:
            output = horizontal_flip(output)
        if dice > 0.7:
            output = vertical_flip(output)

    dice = np.random.randint(0, 10) / 10.0
    if dice < 0.3:
        output = crop(output, crop_bounds)

    dice = np.random.randint(0, 10) / 10.0
    if dice > 0.9:
        output = scale(output, scalex_bound, scaley_bound)

    dice = np.random.randint(0, 10) / 10.0
    if dice <=0.4:
        output= gaussian_blur(output, sigma)

    dice = np.random.randint(0, 10) / 10.0
    if dice >0.5:
        output= contrast(output, gamma)

    dice = np.random.randint(0, 10) / 10.0
    if dice > 0.8:
        output=hue_and_saturation(output, hue_range)
    dice = np.random.randint(0, 10) / 10.0
    if dice > 0.7:
        output = gaussian_noise(output, scale_noise)

    return convert_back_to_uint(output)


def vertical_flip(image):
    """
    Perform a vertical flip on the three images given in input
    :param image: The three input images in one matrix 650x650x(3+3+3)
    :return: The three input images in one matrix 650x650x(3+3+3) flipped on a vertical axis
    """
    flipped = np.zeros(image.shape)
    for index, img in enumerate(image):

        for index2 in [0, 1, 2]:
            array = img[:, :, index2]
            flipped[index, :, :, index2] = np.fliplr(array)

    return convert_back_to_uint(flipped)


def horizontal_flip(image):
    """
    Perform a horizontal flip on the three images given as input
    :param image: The three input images in one matrix 650x650x(3+3+3)
    :return: The three input images in one matrix 650x650x(3+3+3) flipped on a horizontal axis
    """

    flipped = np.zeros(image.shape)
    for index, img in enumerate(image):

        for index2 in [0, 1, 2]:
            array = img[:, :, index2]
            flipped[index, :, :, index2] = np.flip(array, 1)
    return convert_back_to_uint(flipped)


def shear(image, shear_bounds):
    """
    Shear the images
    :param image: RGB matrix of 3 input images
    :param shear_bounds: tuple with min and max bound of the shearing
    :return: RGB matrix of the sheared images
    """
    shear_value = np.random.randint(shear_bounds[0], shear_bounds[1])
    seq = iaa.Sequential([iaa.Affine(shear=(shear_value, shear_value))])
    images_aug = seq(images=image)
    return convert_back_to_uint(images_aug)


def gaussian_blur(image, strength):
    """
    Blur an image with a Gautian Blur
    :param image: the image(s) to be blurred. Can be multiple, blurring is independent of the number of images
    :param strength: tuple contain min valu and max value of the sigma. True value will be random between values.
    :return: RGB matrix of the input with blur
    """
    res = image.copy()
    gauss = iaa.Sequential([iaa.GaussianBlur(sigma=strength)])
    images_to_modified = image[0:2]
    res[0:2] = convert_back_to_uint(gauss(images=images_to_modified))
    return res


def hue_and_saturation(image, hue_range):
    """
    Change hue and saturation by value in range given
    :param image: RGB matrix of the two images. (uselles augmentation on a mask)
    :param hue_range: range of hue and saturation modification (-255, 255)
    :return: RGB matrix
    """
    res = image.copy()
    modifier = iaa.AddToHueAndSaturation(hue_range, per_channel=True)
    res[0:2] = modifier(images=convert_back_to_uint(image[0:2]))
    return res


def rotate(images, bounds):
    """
    Rotate the image by a random angle between the min and max value of the bounds given
    :param images: RGB matrix of all images
    :param bounds: min, max value of the rotating angle
    :return:  RGB matrixes of all images rotated
    """
    rotate_value = np.random.randint(bounds[0], bounds[1])
    rotating = iaa.Affine(rotate=(rotate_value, rotate_value))
    rotated_image = rotating(images=images)
    return convert_back_to_uint(rotated_image)


def gaussian_noise(images, bounds):
    """
    adding gaussian noise to images. Different noise for each image.
    :param images: RGB matrix of two input images. Nois on mask is useless
    :param bounds: min and max of the gaussian noise
    :return: RGB matrix with gaussian Noise
    """
    noisy_images = images.copy()
    gaussian_noise_adder = iaa.AdditiveGaussianNoise(loc=0, scale=bounds, per_channel=0.5)
    noisy_images[0:2] = gaussian_noise_adder(images=convert_back_to_uint(images[0:2]))
    return noisy_images


def crop(images, bounds):
    """

    :param images: RGB matrixes of input images including mask
    :param bounds: min and max of percetnage of cropping
    :return:
    """
    crop_value = np.random.uniform(bounds[0], bounds[1])
    croper = iaa.Crop(percent=(crop_value, crop_value))
    cropped_images = croper(images=images)
    return convert_back_to_uint(cropped_images)


def contrast(images, gamma):
    """
    Modifie the contrast of an image by a gamma distribution.
    :param images: RGB matrixes of two images. contrast on mask is useless
    :param gamma: value of gamma parameter (constraint to 0.8 and 1.05) If too high or low, the image is either black or
     overly saturated
    :return: RBG matrix with contrast changed
    """
    contrasted_images = images.copy()
    contraster = iaa.GammaContrast(gamma=gamma)
    contrasted_images[0:2] = contraster(images=images[0:2])
    return convert_back_to_uint(contrasted_images)


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
    scaler = iaa.Affine(scale={"x": (scalex, scalex), "y": (scaley, scaley)})
    scaled_images = scaler(images=images)
    return convert_back_to_uint(scaled_images)


def convert_back_to_uint(matrix):
    """
    Helper functiun to turn a matrix back into a uint matrix. Applied to the RGB matrixes after modification. Changes
    applied ususally result into a float matrix.
    :param matrix: A 650 x650 matrix of float
    :return: the matrix with all value rounded to uint.
    """
    matrix = matrix.astype(np.float64) / 255
    data = 255 * matrix
    img = data.astype(np.uint8)
    return img
