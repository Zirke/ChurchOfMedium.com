import tensorflow as tf
import numpy as np


def rotate_image_90(image_arr, label_array):
    flipped_images, labels = [], []
    for image in image_arr:
        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy().reshape(299, 299)
        image = np.rot90(image)
        flipped_images.append(image)
    for label in label_array:
        labels.append(label)

    return flipped_images, labels


def flip_image(image_arr, label_array):
    flipped_images, labels = [], []
    for image in image_arr:
        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy().reshape(299, 299)
        image = np.fliplr(image)
        flipped_images.append(image)
    for label in label_array:
        labels.append(label)

    return flipped_images, labels


def one_rotate_image(image):
    if not isinstance(image, (np.ndarray, np.generic)):
        image = image.numpy().reshape(299, 299)
        image = np.rot90(image)

    return image


def one_flip_image(image):
    if not isinstance(image, (np.ndarray, np.generic)):
        image = image.numpy().reshape(299, 299)
        image = np.fliplr(image)

    return image


def produce_more_data(image_array, label_array):
    rotated_images90, rotated90_lbls = rotate_image_90(image_array, label_array)
    rotated_images180, rotated180_lbls = rotate_image_90(rotated_images90, rotated90_lbls)
    rotated_images270, rotated270_lbls = rotate_image_90(rotated_images180, rotated180_lbls)

    image_array = append_arr(image_array, rotated_images90)
    image_array = append_arr(image_array, rotated_images180)
    image_array = append_arr(image_array, rotated_images270)

    label_array = append_labels(label_array, rotated90_lbls)
    label_array = append_labels(label_array, rotated180_lbls)
    label_array = append_labels(label_array, rotated270_lbls)

    rot_flip_images, rot_flip_labels = flip_image(image_array, label_array)
    image_array = append_arr(image_array, rot_flip_images)
    label_array = append_arr(label_array, rot_flip_labels)

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
