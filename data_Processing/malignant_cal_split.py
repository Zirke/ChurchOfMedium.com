import tensorflow as tf

from data_Processing.processing import shuffle
from data_Processing.data_augmentation import *
"""
Purpose of this file is to find the malignant calcification in the original dataset and
create a binary split between malignant calcification and the other categories. 

As of right now the split is 1/3rd MALIGNANT CALCIFICATION and 2/3rd OTHERS
of the 2/3rd only 30% of 1/3rd can be negative
"""


def malignant_cal_split(parsed_data):
    image_array, label_array = length_and_malignant_arrays(parsed_data)
    image_array, label_array = produce_more_data(image_array, label_array)
    non_malignant_imgs, non_malignant_lbls = non_malignant_images(parsed_data, len(image_array))
    non_malignant_imgs, non_malignant_lbls = shuffle(non_malignant_imgs, non_malignant_lbls, len(non_malignant_imgs))
    for image in non_malignant_imgs:
        image_array.append(image)
    for label in non_malignant_lbls:
        label_array.append(label)

    return image_array, label_array


def length_and_malignant_arrays(parsed_data):
    malignant_images, malignant_labels = [], []

    for image, label in parsed_data:
        # 3 is label number is the original dataset for malignant mass
        if label.numpy() == 3:
            indicator_variable = tf.convert_to_tensor([0, 1],
                                                      dtype=tf.int64)  # convert label 3 into 1 for binary classification
            malignant_images.append(image)
            malignant_labels.append(indicator_variable)

    return malignant_images, malignant_labels


def non_malignant_images(parsed_data, n):
    non_malignant_image, non_malignant_labels = [], []
    negative_count = 0

    for image, label in parsed_data:
        # Twice as many non-malignant calcification as malignant calcification
        if len(non_malignant_labels) == n * 2:
            return non_malignant_image, non_malignant_labels
        elif label.numpy() == 0 and negative_count < n // 100 * 30:
            indicator_variable = tf.convert_to_tensor([1, 0], dtype=tf.int64)
            non_malignant_image.append(image)
            non_malignant_labels.append(indicator_variable)
            negative_count += 1
        # 3 is label number is the original dataset for malignant mass
        elif 3 != label.numpy() != 0:
            indicator_variable = tf.convert_to_tensor([1, 0], dtype=tf.int64)  # All other labels should be 0
            non_malignant_image.append(image)
            non_malignant_image.append(one_flip_image(image))
            non_malignant_image.append(one_rotate_image(image))
            non_malignant_labels.append(indicator_variable)
            non_malignant_labels.append(indicator_variable)
            non_malignant_labels.append(indicator_variable)

    return non_malignant_image, non_malignant_labels
