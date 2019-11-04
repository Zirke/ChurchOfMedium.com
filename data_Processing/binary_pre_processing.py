from data_Processing.pre_processing import *

"""
This file extracts information in files with binary classification.

To use it call process_data with a list of file paths for the data extraction to take place.
All file paths can be found in sorting_hub
"""

features_description = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.string, default_value='')
}


def process_data(file_paths):
    extracted_train_data = tf.data.TFRecordDataset(file_paths[0])
    extracted_val_data = tf.data.TFRecordDataset(file_paths[1])
    extracted_test_data = tf.data.TFRecordDataset(file_paths[2])
    return extracted_train_data.map(decode_indicator_variables), extracted_val_data.map(
        decode_indicator_variables), extracted_test_data.map(decode_indicator_variables)


def decode_indicator_variables(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, features_description)
    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = tf.io.decode_raw(feature['label'], tf.int64)

    image = tf.reshape(image, [299, 299, 1])
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label
