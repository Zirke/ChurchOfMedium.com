import os

import cv2
import tensorflow as tf


class TFRecordExtractor:

    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)
        self.count = 0

    def _extract_fn(self, tfrecord):

        # Extract features using the keys set during creation
        feature = {'label': tf.io.FixedLenFeature([], tf.int64),
                   'label_normal': tf.io.FixedLenFeature([], tf.int64),
                   'image': tf.io.FixedLenFeature([], tf.string)}

        # Decode the record read by the reader
        features = tf.io.parse_single_example(tfrecord, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.io.decode_raw(features['image'], tf.uint8)

        label = features['label']

        label_normal = features['label_normal']

        image = tf.reshape(image, [299, 299, 1])

        return [image, label, label_normal]

    def post_process_images(self):

        image_data_list = self.get_images()

        for image_data in image_data_list:
            self.count = self.count + 1

            file_name = 'images/extracted_images/mammography_' + str(image_data[2]) + '_' + str(
                self.count)

            cv2.imwrite(file_name + ".png", image_data[0])

    def get_images(self):
        # Initialize all tfrecord paths

        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            image_data_list = []
            try:
                while True:
                    image_data = sess.run(next_element)
                    image_data_list.append(image_data)

            except:
                pass

            return image_data_list


test = TFRecordExtractor('training10_1/training10_1.tfrecords')
test.post_process_images()
