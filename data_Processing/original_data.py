import tensorflow as tf

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
    image = image / 255.0
    return image, label

def original_dataset():
    path_files = ['training10_0/training10_0.tfrecords',
                  'training10_1/training10_1.tfrecords',
                  'training10_2/training10_2.tfrecords',
                  'training10_3/training10_3.tfrecords',
                  'training10_4/training10_4.tfrecords'
                  ]

    # Extract data as tfrecord dataset
    extracted_data = tf.data.TFRecordDataset(path_files)

    parsed_data = extracted_data.map(decode)

    DATASET_SIZE = len(list(parsed_data))

    training_size = int(0.9 * DATASET_SIZE)
    test_size = int(0.05 * DATASET_SIZE)
    val_size = int(0.05 * DATASET_SIZE)

    training_data_set = parsed_data.take(training_size)
    testing_data_set = parsed_data.skip(training_size)
    validation_data_set = testing_data_set.skip(val_size)
    testing_data_set = testing_data_set.take(test_size)

    return training_data_set, testing_data_set, validation_data_set