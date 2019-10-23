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
            positive_images.append(image)
            positive_labels.append(label)

    return positive_images, positive_labels


def negative_images_arr(parsed_data, n):
    negative_images, negative_labels = [], []
    negative_count = 0

    for image, label in parsed_data:
        if len(negative_labels) == (n // 100) * 3:
            return negative_images, negative_labels
        elif label.numpy() == 0:
            negative_images.append(image)
            negative_labels.append(label)
            negative_count += 1

    return negative_images, negative_labels
