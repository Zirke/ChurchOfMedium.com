import tensorflow as tf
from Data_Processing.pre_processing import *
import numpy as np

feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_normal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.io.FixedLenFeature([], tf.string, default_value='')
}

features_binary = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.io.FixedLenFeature([], tf.string, default_value='')
}


def get_full_dataset():
    path_files = ['training10_0/training10_0.tfrecords',
                  'training10_1/training10_1.tfrecords',
                  'training10_2/training10_2.tfrecords',
                  'training10_3/training10_3.tfrecords',
                  'training10_4/training10_4.tfrecords'
                  ]
    return tf.data.TFRecordDataset(path_files)


def negative_binary_classification():
    dataset = get_full_dataset()
    parsed_data = dataset.map(decode)

    image_array, label_array, n_non_neg = length_and_non_negative_arrays(parsed_data)
    negative_images, negative_labels = negative_image_array(parsed_data, n_non_neg)

    for image in negative_images:
        image_array.append(image)
    for label in negative_labels:
        label_array.append(label)

    features_dataset = tf.data.Dataset.from_tensor_slices((image_array, label_array))
    serialized_dataset = features_dataset.map(tf_serialize_example)

    def generator():
        for features in features_dataset:
            yield serialize_example(*features)

    serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())

    filename = 'negative_binary.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_dataset)


def length_and_non_negative_arrays(parsed_data):
    non_negative_image, non_negative_label = [], []

    for image, label in parsed_data:
        if label.numpy() != 0:
            non_negative_image.append(image)
            non_negative_label.append(label)

    return non_negative_image, non_negative_label, len(non_negative_image)


def negative_image_array(parsed_data, n):
    negative_images, negative_labels = [], []

    for image, label in parsed_data:
        if len(negative_images) == n:
            return negative_images, negative_labels
        elif label.numpy() == 0:
            negative_images.append(image)
            negative_labels.append(label)

    return negative_images, negative_labels


def tf_serialize_example(image, label):
    tf_string = tf.py_function(
        serialize_example,
        (image, label),  # pass these args to the above function.
        tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar


# Creates a tf.Example message ready to be written to a file.
def serialize_example(image, label):
    feature = {
        'image': _float_feature(image),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.numpy().reshape(-1)))
