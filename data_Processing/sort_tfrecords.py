from data_Processing.benign_cal_split import *
from data_Processing.benign_mass_split import *
from data_Processing.malignant_cal_split import *
from data_Processing.malignant_mass_split import *
from data_Processing.negative_sort import *
from data_Processing.pre_processing import *
from data_Processing.five_diagnosis_labels import *
from data_Processing.data_augmentation import *

"""
Purpose of this file is to read the ORIGINAL tfrecords and produce new ones based on sorting. 

For each sorting 3 files are created, a training, validation and test file in tfrecord format. 
All data is serialized and images are converted into bytelists. 

The validation and test files each consists of 5% of the training dataset 

These methods are called from within the sorting_hub mostly. 
"""

SPLIT_SIZE = 15000

feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_normal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
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


# Decode original dataset looking at binary labelling, performing no reshaping or similar
def decode_low(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label_normal']

    return image, label


# Decode the original dataset with full labelling of all classes, performs no reshaping of images.
def decode_low_wide(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label']

    return image, label


# Main method for writing tfrecords. Requires the destined file location as well as the name of the sorting algorithm.
def dataset_sort(file_paths, sorting_algorithm):
    dataset = get_full_dataset()

    image_array, label_array = [], []

    if sorting_algorithm == 'negative_bi':
        parsed_data = dataset.map(decode_low)
        image_array, label_array = negative_bi_split(parsed_data)

    elif sorting_algorithm == 'benign_cal_split':
        parsed_data = dataset.map(decode_low_wide)
        image_array, label_array = benign_cal_split(parsed_data)

    elif sorting_algorithm == 'benign_mass_split':
        parsed_data = dataset.map(decode_low_wide)
        image_array, label_array = benign_mass_split(parsed_data)

    elif sorting_algorithm == 'malignant_cal_split':
        parsed_data = dataset.map(decode_low_wide)
        image_array, label_array = malignant_cal_split(parsed_data)

    elif sorting_algorithm == 'malignant_mass_split':
        parsed_data = dataset.map(decode_low_wide)
        image_array, label_array = malignant_mass_split(parsed_data)

    elif sorting_algorithm == 'five_diagnosis':
        parsed_data = dataset.map(decode_low_wide)
        image_array, label_array = append_arrays(parsed_data)

    with tf.device('/CPU:0'):
        amount_of_imgs = len(image_array)
        validation_images, validation_labels, image_array, label_array = five_percent_to_arrays(image_array,
                                                                                                label_array,
                                                                                                amount_of_imgs)
        validation_images, validation_labels = shuffle(validation_images, validation_labels, len(validation_images))
        write_to_tf(file_paths, validation_images, validation_labels, 'validation')

        testing_images, testing_labels, image_array, label_array = five_percent_to_arrays(image_array, label_array,
                                                                                          amount_of_imgs)

        testing_images, testing_labels = shuffle(testing_images, testing_labels, len(testing_images))
        write_to_tf(file_paths, testing_images, testing_labels, 'test')

        image_array, label_array = shuffle(image_array, label_array, len(image_array))

        write_to_tf(file_paths, image_array, label_array, 'training')


# write dataset into tfrecord
def write_to_tf(file_paths, image_array, label_array, write_type):
    image = list(chunks_of_array(image_array, SPLIT_SIZE))
    label = list(chunks_of_array(label_array, SPLIT_SIZE))

    filenr = 0
    with tf.device('/CPU:0'):
        for x in image:
            dataset = tf.data.Dataset.from_tensor_slices((conv_to_tensor(x), label[filenr]))
            serialized_tr_dataset = dataset.map(tf_serialize_example)

            def generator():
                for features in dataset:
                    yield serialize_example(*features)

            serialized_tr_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())

            if write_type == 'training':
                write_file_path = file_paths[0] + str(filenr) + '.tfrecord'
                writer_tr = tf.data.experimental.TFRecordWriter(write_file_path)
                writer_tr.write(serialized_tr_dataset)
                filenr += 1
            elif write_type == 'validation':
                writer_tr = tf.data.experimental.TFRecordWriter(file_paths[1])
                writer_tr.write(serialized_tr_dataset)
            elif write_type == 'test':
                writer_tr = tf.data.experimental.TFRecordWriter(file_paths[2])
                writer_tr.write(serialized_tr_dataset)


# divides array into n chunk arrays
def chunks_of_array(array, n):
    for i in range(0, len(array), n):
        yield array[i:i + n]


# Creates an array of images and labels consisting of 5% of the training data. Used for val and test tfrecords.
def five_percent_to_arrays(image_arr, label_arr, n):
    five_pc_img, five_pc_label = [], []

    for i in range((n // 100) * 5):
        if i % 2 == 0:
            five_pc_img.append(image_arr[i])
            del image_arr[i]
            five_pc_label.append(label_arr[i])
            del label_arr[i]
        else:
            five_pc_img.append(image_arr[len(image_arr) - 1])
            del image_arr[len(image_arr) - 1]
            five_pc_label.append(label_arr[len(label_arr) - 1])
            del label_arr[len(label_arr) - 1]

    return five_pc_img, five_pc_label, image_arr, label_arr


# serialize image and label
def tf_serialize_example(image, label):
    tf_string = tf.py_function(
        serialize_example,
        (image, label),  # pass these args to the above function.
        tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar


# Creates a tf.Example message ready to be written to a file.
def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Creates a int64 feature based on the labels
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Used for creating a bytelist from a picture
def _bytes_feature(value):
    value = value.numpy().tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def conv_to_tensor(image_array):
    tensor_arr = []
    for image in image_array:
        conv = tf.convert_to_tensor(image, dtype=tf.uint8)
        tensor_arr.append(tf.reshape(conv, [-1, 299, 299, 1]))
    return tensor_arr
