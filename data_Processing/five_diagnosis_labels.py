import tensorflow as tf
from data_Processing.processing import shuffle
from data_Processing.data_augmentation import *
"""
Purpose of this file is to create two arrays that consist of images and labels for classification 
of 5 diagnosis types, negative, benign cal, benign mass, malignant cal, malignant mass. 

The sorting is to take all of the 4 positive type categories and turn them into it array, then 
get roughly 30% of the arrays size and append as many negative images to it. 
"""


def append_arrays(parsed_data):
    existing_images, existing_labels = positive_images_arr(parsed_data)
    existing_images, existing_labels = produce_more_data(existing_images, existing_labels)
    app_imgs, app_lbls = negative_images_arr(parsed_data, len(existing_images))

    for image in app_imgs:
        existing_images.append(image)
    for label in app_lbls:
        existing_labels.append(label)

    return shuffle(existing_images, existing_labels, len(existing_images))


def positive_images_arr(parsed_data):
    positive_images, positive_labels = [], []

    for image, label in parsed_data:
        if label.numpy() != 0:
            if label.numpy() == 1:
                indicator_variable = tf.convert_to_tensor([0, 1, 0, 0, 0], dtype=tf.int64)
                positive_images.append(image)
                positive_labels.append(indicator_variable)
            elif label.numpy() == 2:
                indicator_variable = tf.convert_to_tensor([0, 0, 1, 0, 0], dtype=tf.int64)
                positive_images.append(image)
                positive_labels.append(indicator_variable)
            elif label.numpy() == 3:
                indicator_variable = tf.convert_to_tensor([0, 0, 0, 1, 0], dtype=tf.int64)
                positive_images.append(image)
                positive_labels.append(indicator_variable)
            elif label.numpy() == 4:
                indicator_variable = tf.convert_to_tensor([0, 0, 0, 0, 1], dtype=tf.int64)
                positive_images.append(image)
                positive_labels.append(indicator_variable)

    return positive_images, positive_labels


def negative_images_arr(parsed_data, n):
    negative_images, negative_labels = [], []
    negative_count = 0

    for image, label in parsed_data:
        if len(negative_labels) == (n // 100) * 25:
            return negative_images, negative_labels
        elif label.numpy() == 0:
            indicator_variable = tf.convert_to_tensor([1, 0, 0, 0, 0], dtype=tf.int64)
            negative_images.append(image)
            negative_labels.append(indicator_variable)
            negative_count += 1

    return negative_images, negative_labels
