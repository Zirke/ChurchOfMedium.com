import tensorflow as tf
import numpy as np


def flip_image_90(image_arr, label_array):
    flipped_images, labels = [], []
    for image in image_arr:
        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy().reshape(299, 299)
        image = np.rot90(image)
        flipped_images.append(image)
    for label in label_array:
        labels.append(label)

    return flipped_images, labels


def produce_more_data(image_array, label_array):
    flipped_images90, flipped90_lbls = flip_image_90(image_array, label_array)
    flipped_images180, flipped180_lbls = flip_image_90(flipped_images90, flipped90_lbls)
    flipped_images270, flipped270_lbls = flip_image_90(flipped_images180, flipped180_lbls)

    image_array = append_arr(image_array, flipped_images90)
    image_array = append_arr(image_array, flipped_images180)
    image_array = append_arr(image_array, flipped_images270)

    label_array = append_labels(label_array, flipped90_lbls)
    label_array = append_labels(label_array, flipped180_lbls)
    label_array = append_labels(label_array, flipped270_lbls)
    return image_array, label_array


def append_arr(image_array, append_arr):
    for append in append_arr:
        conv = tf.convert_to_tensor(append)
        image_array.append(conv)
    return image_array


def append_labels(label_array, append_label):
    for label in append_label:
        label_array.append(label)
    return label_array
