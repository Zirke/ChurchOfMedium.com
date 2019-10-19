import tensorflow as tf
import matplotlib.pyplot as plt

path_file = ['negative_binary.tfrecord']

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}


def process_data():
    extracted_data = tf.data.TFRecordDataset(path_file)

    return extracted_data.map(decode)


def decode(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label']

    image = tf.reshape(image, [299, 299, 1])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label
