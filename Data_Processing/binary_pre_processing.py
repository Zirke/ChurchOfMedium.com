import tensorflow as tf
import matplotlib.pyplot as plt
from Data_Processing.pre_processing import *

"""
This file extracts information in files with binary classification.

To use it call process_data with a list of file paths for the data extraction to take place.
All file paths can be found in sorting_hub
"""

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}


def process_data(file_paths):
    extracted_train_data = tf.data.TFRecordDataset(file_paths[0])
    extracted_val_data = tf.data.TFRecordDataset(file_paths[1])
    extracted_test_data = tf.data.TFRecordDataset(file_paths[2])
    return extracted_train_data.map(decode), extracted_val_data.map(decode), extracted_test_data.map(decode)
