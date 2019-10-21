import tensorflow as tf
from Data_Processing.processing import shuffle
"""
Purpose of this file is to find the malignant mass in the original dataset and
create a binary split between malignant calcification and the other categories. 

As of right now the split is 1/3rd MALIGNANT MASS and 2/3rd OTHERS
of the 2/3rd only 20% of 1/3rd can be negative
"""


def malignant_mass_split(parsed_data):
    image_array, label_array, n_malignant = length_and_malignant_arrays(parsed_data)
    non_malignant_imgs, non_malignant_lbls = non_malignant_images(parsed_data, n_malignant)
    non_malignant_imgs, non_malignant_lbls = shuffle(non_malignant_imgs, non_malignant_lbls, len(non_malignant_imgs))
    for image in non_malignant_imgs:
        image_array.append(image)
    for label in non_malignant_lbls:
        label_array.append(label)

    return image_array, label_array


def length_and_malignant_arrays(parsed_data):
    malignant_images, malignant_labels = [], []

    for image, label in parsed_data:
        # label 4 is the number in the original dataset
        if label.numpy() == 4:
            label_one = tf.convert_to_tensor(1, dtype=tf.int64)  # convert to label 1 for binary classification
            malignant_images.append(image)
            malignant_labels.append(label_one)

    return malignant_images, malignant_labels, len(malignant_images)


def non_malignant_images(parsed_data, n):
    non_malignant_image, non_malignant_labels = [], []
    negative_count = 0

    for image, label in parsed_data:
        # Twice as many non-malignant mass as malignant mass
        if len(non_malignant_labels) == n * 2:
            return non_malignant_image, non_malignant_labels
        elif label.numpy() == 0 and negative_count < n // 100 * 20:
            non_malignant_image.append(image)
            non_malignant_labels.append(label)
            negative_count += 1
        # 4 is the label for malignant mass in the original dataset
        elif 4 != label.numpy() != 0:
            zero_lbl = tf.convert_to_tensor(0, dtype=tf.int64)  # All other labels should be 0
            non_malignant_image.append(image)
            non_malignant_labels.append(zero_lbl)

    return non_malignant_image, non_malignant_labels
