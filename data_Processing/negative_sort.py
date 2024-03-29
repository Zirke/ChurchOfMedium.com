"""
Purpose of this file is to find the negatives in the original dataset and
create a binary split between negatives and the other categories.

As of right now the split is 1/2 NEGATIVES and 1/2 OTHERS
"""
from data_Processing.processing import shuffle
from data_Processing.data_augmentation import *

def negative_bi_split(parsed_data):
    image_array, label_array = length_and_non_negative_arrays(parsed_data)
    image_array, label_array = produce_more_data(image_array, label_array)
    negative_images, negative_labels = negative_image_array(parsed_data, len(image_array))
    negative_images, negative_labels = shuffle(negative_images, negative_labels, len(negative_images))
    for image in negative_images:
        image_array.append(image)
    for label in negative_labels:
        label_array.append(label)

    return image_array, label_array


def length_and_non_negative_arrays(parsed_data):
    non_negative_image, non_negative_label = [], []

    for image, label in parsed_data:
        # In the original dataset 0 is negatives
        if label.numpy() != 0:
            indicator_variable = tf.convert_to_tensor([0, 1], dtype=tf.int64)
            non_negative_image.append(image)
            non_negative_label.append(indicator_variable)

    return non_negative_image, non_negative_label


def negative_image_array(parsed_data, n):
    negative_images, negative_labels = [], []

    for image, label in parsed_data:
        # Fill up the array with as many others as negatives
        if len(negative_images) == n:
            return negative_images, negative_labels
        elif label.numpy() == 0:
            indicator_variable = tf.convert_to_tensor([1, 0], dtype=tf.int64)
            negative_images.append(image)
            negative_labels.append(indicator_variable)

    return negative_images, negative_labels
