import tensorflow as tf

files = ['training10_0/training10_0.tfrecords']

extracted_data = tf.data.TFRecordDataset(files)

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

def get_dataset():
    return extracted_data.map(decode)