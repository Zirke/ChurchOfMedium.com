import tensorflow as tf

from data_Processing.processing import shuffle
from data_Processing.data_augmentation import *
"""
Purpose of this file is to find the benign mass in the original dataset and
create a binary split between benign mass and the other categories. 

As of right now the split is 1/3rd BENIGN MASS and 2/3rd OTHERS
of the 2/3rd only 30% of 1/3rd can be negative
"""


def benign_mass_split(parsed_data):
    image_array, label_array = length_and_benign_arrays(parsed_data)
    image_array, label_array = produce_more_data(image_array, label_array)
    non_benign_imgs, non_benign_lbls = non_benign_images(parsed_data, len(image_array))
    non_benign_imgs,non_benign_lbls = shuffle(non_benign_imgs,non_benign_lbls, len(non_benign_imgs))
    for image in non_benign_imgs:
        image_array.append(image)
    for label in non_benign_lbls:
        label_array.append(label)

    return image_array, label_array


def length_and_benign_arrays(parsed_data):
    benign_images, benign_labels = [], []

    for image, label in parsed_data:
        # 2 is label number is the original dataset for benign mass
        if label.numpy() == 2:
            indicator_variable = tf.convert_to_tensor([0, 1], dtype=tf.int64)  # In this sorted list it's converted to 1
            benign_images.append(image)
            benign_labels.append(indicator_variable)

    return benign_images, benign_labels


def non_benign_images(parsed_data, n):
    non_benign_image, non_benign_labels = [], []
    negative_count = 0

    for image, label in parsed_data:
        # Twice as many non-benign mass as benign mass
        if len(non_benign_labels) == n * 2:
            return non_benign_image, non_benign_labels
        # We only want a certain amount of negatives
        elif label.numpy() == 0 and negative_count < n // 100 * 30:
            indicator_variable = tf.convert_to_tensor([1, 0], dtype=tf.int64)
            non_benign_image.append(image)
            non_benign_labels.append(indicator_variable)
            negative_count += 1
        # 2 is label number is the original dataset for benign mass
        elif 2 != label.numpy() != 0:
            indicator_variable = tf.convert_to_tensor([1, 0], dtype=tf.int64)  # All other labels should be 0
            non_benign_image.append(image)
            non_benign_image.append(one_flip_image(image))
            non_benign_image.append(one_rotate_image(image))
            non_benign_labels.append(indicator_variable)
            non_benign_labels.append(indicator_variable)
            non_benign_labels.append(indicator_variable)

    return non_benign_image, non_benign_labels
