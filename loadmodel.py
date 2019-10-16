from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(299, 299, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


test_path_file = ['training10_4/training10_4.tfrecords']
extracted_test_data = tf.data.TFRecordDataset(test_path_file)

feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_normal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.io.FixedLenFeature([], tf.string, default_value='')
}


def decode(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label']

    image = tf.reshape(image, [299, 299, 1])
    image = tf.cast(image, tf.float32)
    return image, label


BATCH_SIZE = 32
parsed_testing_data = extracted_test_data.map(decode)
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()

model = create_model()
checkpoint_path = "training_1/cp.ckpt"

model.load_weights(checkpoint_path)

results = model.evaluate(batched_testing_data, steps=300 // BATCH_SIZE)
print('test loss, test acc:', results)
