"""
Purpose of this file is to find the negatives in the original dataset and
create a binary split between negatives and the other categories.

As of right now the split is 1/2 NEGATIVES and 1/2 OTHERS
"""


def negative_bi_split(parsed_data):
    image_array, label_array, n_non_neg = length_and_non_negative_arrays(parsed_data)
    negative_images, negative_labels = negative_image_array(parsed_data, n_non_neg)

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
            non_negative_image.append(image)
            non_negative_label.append(label)

    return non_negative_image, non_negative_label, len(non_negative_image)


def negative_image_array(parsed_data, n):
    negative_images, negative_labels = [], []

    for image, label in parsed_data:
        # Fill up the array with as many others as negatives
        if len(negative_images) == n:
            return negative_images, negative_labels
        elif label.numpy() == 0:
            negative_images.append(image)
            negative_labels.append(label)

    return negative_images, negative_labels
