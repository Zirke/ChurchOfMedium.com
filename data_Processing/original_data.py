import tensorflow as tf

feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_normal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.io.FixedLenFeature([], tf.string, default_value='')
}

SPLIT_SIZE = 15000


def decode(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label']

    image = tf.reshape(image, [299, 299, 1])
    #image = tf.cast(image, tf.float32)
    #image = image / 255.0
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

    image_array, label_array = indicator_variables(parsed_data)

    DATASET_SIZE = len(list(image_array))

    training_size = int(0.9 * DATASET_SIZE)
    test_size = int(0.05 * DATASET_SIZE)
    val_size = int(0.05 * DATASET_SIZE)

    validation_images, validation_labels, image_array, label_array = five_percent_to_arrays(image_array,
                                                                                            label_array,
                                                                                            DATASET_SIZE)
    write_to_tf(['sorted_tfrecords/original_indicator/validation.tfrecord'], validation_images, validation_labels, 'validation')

    testing_images, testing_labels, image_array, label_array = five_percent_to_arrays(image_array, label_array,
                                                                                      DATASET_SIZE)

    write_to_tf(['sorted_tfrecords/original_indicator/test.tfrecord'], testing_images, testing_labels, 'test')

    write_to_tf(['sorted_tfrecords/original_indicator/training'], image_array, label_array, 'training')


def indicator_variables(parsed_data):
    all_images, all_labels = [], []

    for image, label in parsed_data:
        if label.numpy() != 0:
            if label.numpy() == 1:
                indicator_variable = tf.convert_to_tensor([0, 1, 0, 0, 0], dtype=tf.int64)
                all_images.append(image)
                all_labels.append(indicator_variable)
            elif label.numpy() == 2:
                indicator_variable = tf.convert_to_tensor([0, 0, 1, 0, 0], dtype=tf.int64)
                all_images.append(image)
                all_labels.append(indicator_variable)
            elif label.numpy() == 3:
                indicator_variable = tf.convert_to_tensor([0, 0, 0, 1, 0], dtype=tf.int64)
                all_images.append(image)
                all_labels.append(indicator_variable)
            elif label.numpy() == 4:
                indicator_variable = tf.convert_to_tensor([0, 0, 0, 0, 1], dtype=tf.int64)
                all_images.append(image)
                all_labels.append(indicator_variable)

        elif label.numpy() == 0:
            indicator_variable = tf.convert_to_tensor([1, 0, 0, 0, 0], dtype=tf.int64)
            all_images.append(image)
            all_labels.append(indicator_variable)

    return all_images, all_labels


def write_to_tf(file_paths, image_array, label_array, write_type):
    image = list(chunks_of_array(image_array, SPLIT_SIZE))
    label = list(chunks_of_array(label_array, SPLIT_SIZE))

    filenr = 0
    with tf.device('/CPU:0'):
        for x in image:
            dataset = tf.data.Dataset.from_tensor_slices((x, label[filenr]))
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
                writer_tr = tf.data.experimental.TFRecordWriter(file_paths[0])
                writer_tr.write(serialized_tr_dataset)
            elif write_type == 'test':
                writer_tr = tf.data.experimental.TFRecordWriter(file_paths[0])
                writer_tr.write(serialized_tr_dataset)


# divides array into n chunk arrays
def chunks_of_array(array, n):
    for i in range(0, len(array), n):
        yield array[i:i + n]


def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Used for creating a bytelist from a picture
def _bytes_feature(value):
    value = value.numpy().tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_serialize_example(image, label):
    tf_string = tf.py_function(
        serialize_example,
        (image, label),  # pass these args to the above function.
        tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar


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
