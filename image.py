import numpy as np
from PIL import Image
import imageio
import torch
from augmentation import *
import os
import matplotlib.pyplot as plt


# Hardcoded instance numbers
# For the Earth Dataset, we have 32 instances from 1 to 32
IMAGE_NUM = list(range(1, 33))


""" Loading the dataset """


def load_dataset(img_nums, n_augmentation_per_image, batch_size=1):
    """
    Loads images from the Earth dataset, applies the necessary preprocessing and put them into a dataloader format
    :param img_nums: list of numbers of the instance we want to load
    :param n_augmentation_per_image: the number of augmented instances to generate per instance, use 0 to not perform data_augmentation
    :param batch_size: number of instances per batch
    :return: a dataloader object containing the dataset
    """
    no_augment = False
    if n_augmentation_per_image == 0:
        no_augment = True
        n_augmentation_per_image = 1

    inputs = np.zeros(shape=((n_augmentation_per_image) * len(img_nums), 6, 650, 650))
    masks  = np.zeros(shape=((n_augmentation_per_image) * len(img_nums), 650, 650), dtype=np.long)

    for i, img_num in enumerate(img_nums):
        # Opening
        img_b = open_image("DATA/Earth_" + str(img_num) + "/before.png")
        img_a = open_image("DATA/Earth_" + str(img_num) + "/after.png")
        img_m = open_image("DATA/Earth_" + str(img_num) + "/mask.png")

        # Prepare
        input, mask = images_prepare(img_b, img_a, img_m)

        # Augmentation
        augmentedData = None
        if not no_augment :
            augmentedData = data_augmentation(img_a, img_b, img_m, n_augmentation_per_image)

        # Storing
        j = i * (n_augmentation_per_image)
        inputs[j] = input
        masks[j] = mask
        if not no_augment:
            for l in range(0,len(augmentedData)):
                inputs[j+l] = augmentedData[l][0]
                masks[j+l] = augmentedData[l][1]

        inputs[j] = input
        masks[j] = mask

    # Batching
    inputs, masks = fold_batch(inputs, masks, batch_size)
    return dataset_to_dataloader(inputs, masks)


def load_dataset_predict(input_dir, output_dir, instance_names, batch_size=1):
    """
    Loads images from the Earth dataset, applies the necessary preprocessing and put them into a dataloader format
    This version is to be used by predict
    :param input_dir: path to the directory where the instances are located
    :param output_dir: path to the directory where to save the output masks
    :param instance_names: list of directory names
    :param batch_size: number of instances per batch
    :return: a dataloader object containing the dataset
    """

    inputs = np.zeros(shape=(len(instance_names), 6, 650, 650))
    masks = np.zeros(shape=(len(instance_names), 650, 650), dtype=np.long)
    output_paths = []

    i = 0
    for inst in instance_names:
        try:
            # Opening
            img_b = open_image(input_dir + "/" + inst + "/before.png")
            img_a = open_image(input_dir + "/" + inst + "/after.png")
            img_m = np.zeros(shape=(650, 650))

            # Prepare
            input, mask = images_prepare(img_b, img_a, img_m)

            inputs[i] = input
            masks[i] = mask
            output_paths.append(output_dir + "/" + inst)
            i += 1
        except Exception as e:
            #print("WARNING : error while opening instance " + input_dir + inst)
            print(e)
    inputs = inputs[:i]
    masks = masks[:i]

    # Batching
    inputs, masks = fold_batch(inputs, masks, batch_size)
    return dataset_to_dataloader(inputs, masks), output_paths


def data_augmentation(before, after, mask, n_augmentation):
    """
    Applies data augmentation on instances
    :param before: the "before" image of an instance
    :param after: the "after" image of an instance
    :param mask: the ground truth mask of an instance
    :param n_augmentation: number of different augmented instances to generate
    :return: a list of augmented instances
    """
    augmentedData = []
    input = np.zeros((3, 650, 650, 3))

    input[0] = before[..., [0,1,2]]
    input[1] = after[..., [0,1,2]]
    input[2] = mask[..., [0,1,2]]

    for i in range(n_augmentation) :
        [im_a,im_b,mask_c] = applyAugmentation(input)
        augmentedData.append(images_prepare(im_b, im_a, mask_c))

    return augmentedData


def images_prepare(img_before, img_after, img_mask):
    """
    Sub-function of load_dataset
    Merges the before and after images, applies the necessary transformation on the three given images
    :param img_before: the "before" image of an instance
    :param img_after: the "after" image of an instance
    :param img_mask: the ground truth mask of an instance
    :return: i_join : the before and after images merged, i_m : the processed mask
    """
    i_b = img_before[..., [0, 1, 2]]
    i_a = img_after[..., [0, 1, 2]]
    i_b = normalize(i_b)
    i_a = normalize(i_a)
    i_m = grey_split(img_mask)
    i_m_reverse = reverse_mask(i_m)
    i_join = np.zeros(shape=(6, i_b.shape[0], i_b.shape[1]))
    i_join[[0, 1, 2], ...] = np.transpose(i_b[...], axes=(2, 0, 1))
    i_join[[3, 4, 5], ...] = np.transpose(i_a[...], axes=(2, 0, 1))
    i_m_join = np.zeros(shape = (2, i_m.shape[0], i_m.shape[1]))
    i_m_join[0,...] = i_m
    i_m_join[1,...] = i_m_reverse
    return i_join, i_m


def dataset_to_dataloader(inputs, masks):
    """
    Converts a dataset to the PyTorch dataloader format
    :param inputs: list of input image pairs
    :param masks: list of input ground truth masks
    :return: a dataloader containing the dataset
    """
    tensor_x = torch.stack([torch.Tensor(i) for i in inputs])
    tensor_y = torch.stack([torch.tensor(i, dtype=torch.long) for i in masks])

    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    my_dataloader = torch.utils.data.DataLoader(my_dataset)

    return my_dataloader


""" IO """


def placeholder_file(path):
    """
    Creates an empty file at the given path if it doesn't already exists
    :param path: relative path of the file to be created
    """
    import os
    if not os.path.exists(path):
        with open(path, 'w'): pass


def placeholder_path(path):
    """
    Creates the directories of a path if it doesn't already exists
    :param path: path of the directories to create
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def open_image(filename):
    """
    opens an image file
    :param filename: the relative path of the image file
    :return: the image in a compatible format
    """
    # Loading a tif file
    if filename[-4:].lower() in [".tif", "tiff"]:
        import tifffile as tif
        return tif.imread(filename)
    # Loading a png file
    else:
        return imageio.imread(filename)


def save_masks(masks_predicted, ground_truths, device, max_img=10, shuffle=False, color="blue", filename="mask_predicted.png", threshold=None):
    """
    Saves multiple ground truths and their prediction on a big single image
    :param masks_predicted: list of predicted images (batched)
    :param ground_truths: list of ground truths (batched)
    :param device: device used to train the model (cpu or cuda)
    :param max_img: maximum number if images to display
    :param shuffle: random image order
    :param color: background color for the predicted masks (blue or red)
    :param filename: in which file we will save the image
    :param threshold: used to find the class of each pixel, should be between 0 and 1
    """

    masks_predicted = unfold_batch(masks_predicted)
    ground_truths = unfold_batch(ground_truths)


    max_img = min(max_img, len(masks_predicted))
    import math
    nrow = min(max_img, 10)
    ncol = int(math.ceil(max_img/10))
    from random import sample
    smp = sample(list(range(len(masks_predicted))), max_img)
    if not shuffle:
        smp = list(range(len(masks_predicted)))[0:max_img]
    out = np.ones((nrow*650, 2*650*ncol, 3), 'uint8')

    for i, n in enumerate(smp):
        ir = i % 10
        ic = int(math.floor(i/10))

        mp = masks_predicted[n]
        gt = ground_truths[n]

        arrs = np.zeros(shape=(2, 650, 650))
        gt_arrs = np.zeros(shape=(650, 650))
        if device == 'cuda':
            arrs[...] = mp.cpu().detach().numpy()[...]
            gt_arrs[...] = gt.cpu().detach().numpy()[...]
        else:
            arrs[...] = mp.detach().numpy()[0, ...]
            gt_arrs[...] = gt.detach().numpy()[0, ...]

        arrs = arrs[0, :, :]
        arrs = 1 - arrs
        arrs[arrs < 0] = 0

        # Threshold
        if threshold is not None:
            arrs[arrs < threshold] = 0
            arrs[arrs >= threshold] = 1

        arrs *= 255

        # PREDICTED

        rgbArray = np.ones((650, 650, 3), 'uint8')
        rgbArray *= 255
        if color != "red":
            rgbArray[..., 0] = arrs
        rgbArray[..., 1] = arrs
        if color != "blue":
            rgbArray[..., 2] = arrs

        # GROUND TRUTH

        gt_rgbArray = np.ones((650, 650, 3), 'uint8')
        gt_rgbArray[..., 0] = gt_arrs
        gt_rgbArray[..., 1] = gt_arrs
        gt_rgbArray[..., 2] = gt_arrs

        gt_rgbArray *= 255

        out[650*ir:650*(ir+1), (2*650*ic):(2*650*ic+650), 0:3] = rgbArray
        out[650*ir:650*(ir+1), (2*650*ic+650):(2*650*(ic+1)), 0:3] = gt_rgbArray

    img = Image.fromarray(out)
    img = img.convert("RGB")
    img.save(filename)


def save_predicted_mask(mask, device, color="red", filename="mask_predicted.png", threshold=None):
    """
    Saves multiple ground truths and their prediction on a big single image
    :param mask: predicted mask to save (batched)
    :param device: device used to predict the mask (cpu or cuda)
    :param color: background color for the predicted masks (blue or red, black otherwise)
    :param filename: in which file we will save the image
    :param threshold: used to find the class of each pixel, should be between 0 and 1
    """

    mask = unfold_batch(mask)

    out = np.ones((650, 650, 3), 'uint8')

    mp = mask[0]

    arrs = np.zeros(shape=(2, 650, 650))
    if str(device) == 'cuda':
        arrs[...] = mp.cpu().detach().numpy()[...]
    else:
        arrs[...] = mp.detach().numpy()[0, ...]

    arrs = arrs[0, :, :]
    arrs = 1 - arrs
    arrs[arrs < 0] = 0

    # Threshold
    if threshold is not None:
        arrs[arrs < threshold] = 0
        arrs[arrs >= threshold] = 1

    arrs *= 255

    # PREDICTED

    rgbArray = np.ones((650, 650, 3), 'uint8')
    rgbArray *= 255
    if color != "red":
        rgbArray[..., 0] = arrs
    rgbArray[..., 1] = arrs
    if color != "blue":
        rgbArray[..., 2] = arrs

    out[0:650, 0:650, 0:3] = rgbArray

    img = Image.fromarray(out)
    img = img.convert("RGB")
    placeholder_file(filename)
    img.save(filename)


""" Image and Mask manipulation """


def normalize(image):
    """
    Independant range normalization on the three color channel of an image
    :param image: the image to normalize as a numpy matrix
    :return: the normalized image
    """

    # R G B, with sometimes a 4th Alpha channel on PNG
    arrs = [None, None, None]
    for i in range(3):

        arr = np.copy(image[...,i])

        # Normalization between 0 and 255
        mx = arr.max()
        arr = arr / (1.0 * mx) * 255

        # Remove outliers and normalize
        hist = np.sort(arr.flatten())
        lo = hist[int(0.025 * len(hist))]
        hi = hist[int(0.975 * len(hist))]
        arr = (arr-lo) / (hi-lo) * 255.0

        # Limiting the min and maximal value
        arr[arr < 0] = 0
        arr[arr > 255] = 255

        arrs[i] = arr

    arrs_np = np.zeros(shape=(650, 650, 3))
    arrs_np[:, :, 0] = arrs[0]
    arrs_np[:, :, 1] = arrs[1]
    arrs_np[:, :, 2] = arrs[2]

    return arrs_np


def mask_to_image(masks):
    """
    From 2 class masks, returns a single mask indicating the class of each pixel (trivial for 2 classes)
    :param masks: a numpy matrix of shape 2x650x650
    :return: a 650x650 mask containing the class at each pixel
    """
    mask = np.zeros((650, 650))
    for i in range(650):
        for j in range(650):
            mask[i, j] = masks[0, i, j]
    return mask


def reverse_mask(mask):
    """
    Switches the classes of a 2-class mask
    :param mask: numpy matrix
    :return: the reversed mask
    """
    reversed_mask = mask.copy()
    reversed_mask = np.where(reversed_mask == 1, 0, 1)
    return reversed_mask


def grey_split(mask):
    """
    Turns a greyscale mask to a 2 colors mask
    :param mask: the greyscale input mask
    :return: a 2 colors masks
    """
    grey = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.long)
    grey[...] = mask[..., 0]
    grey[grey != 0] = 1.0
    return grey


""" Batching """


def fold_batch(inputs, masks, batch_size):
    """
    Merge several inputs and their corresponding ground truths masks to create batches
    :param inputs: list of numpy matrices of shape 6x650x650
    :param masks: list of numpy matrices of shape 650x650
    :param batch_size: the desired batch_size
    :return: new_inputs, new_masks : lists of batched inputs and masks
    """

    new_inputs = []
    new_masks = []

    remaining = len(inputs)
    next_pick = min(remaining, batch_size)
    idx = 0

    while remaining > 0:
        batch_inputs = np.stack(inputs[idx:idx+next_pick], axis=0)
        batch_masks = np.stack(masks[idx:idx+next_pick], axis=0)

        new_inputs.append(batch_inputs)
        new_masks.append(batch_masks)

        idx += next_pick
        remaining -= next_pick
        next_pick = min(remaining, batch_size)

    return new_inputs, new_masks


def unfold_batch(batch_list):
    """
    Unfolds a list of batched masks into a list of individual masks
    :param batch_list: a list of batched masks
    :return: a list of individual masks
    """
    ret = []
    for batch in batch_list:
        for i in range(batch.shape[0]):
            ret.append(batch[i, ...])
    return ret

