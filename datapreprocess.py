import tensorflow as tf

# Each dataset has a description of the features within it.
feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'label_normal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image': tf.io.FixedLenFeature([], tf.string, default_value='')
    }

def process_dataset():
    path_files = ['training10_0/training10_0.tfrecords',
                        'training10_1/training10_1.tfrecords',
                        'training10_2/training10_2.tfrecords',
                        'training10_3/training10_3.tfrecords',
                        'training10_4/training10_4.tfrecords'
                        ]

    # Extract data as tfrecord dataset
    extracted_data = tf.data.TFRecordDataset(path_files)

    parsed_data = extracted_data.map(decode)

    t_image, t_label, v_image, v_label, te_image, te_label = [],[],[],[],[],[]
    training_data= dataset_with_same_amount(parsed_data, get_lowest_size(parsed_data), True)
    processed_training_data, processed_val_data,processed_test_data = None,None,None
    count = 0
    for image,label in training_data:
        #For validation test
        if count <= 500:
            v_image.append(image)
            v_label.append(label)
        elif count > 500 and count <= 1000:
            te_image.append(image)
            te_label.append(label)
        else:
            t_image.append(image)
            t_label.append(label)
        count += 1

    #conversion to dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((t_image, t_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((te_image, te_label))
    val_dataset = tf.data.Dataset.from_tensor_slices((v_image, v_label))
    return (training_data,val_dataset,test_dataset)


def _parse_function(example_proto):
   return tf.io.parse_single_example(example_proto, feature_description)


def decode(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label']

    image = tf.reshape(image, [299, 299, 1])
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label

def dataset_with_same_amount(dataset, sizeOfEach, return_data_or_amount):
    count =0
    labels = [0,0,0,0,0]
    #zero, one ,two, three, four = 0
    zero_indices  = []
    one_indices  = []
    two_indices  = []
    three_indices = []
    four_indices = []

    varied_dataset_image = []
    varied_dataset_label = []
    #Zero: 9751, One: 424, Two: 346, Three 287, Four: 369
    for image, label in dataset:
        if label.numpy() == 0 and count > 0:
            labels[0] += 1
            if sizeOfEach > labels[0]:
                varied_dataset_image.append(image)
                varied_dataset_label.append(label)
                zero_indices.append(count)
        elif label.numpy() == 1:
            labels[1] += 1
            if sizeOfEach > labels[1]:
                varied_dataset_image.append(image)
                varied_dataset_label.append(label)
                one_indices.append(count)
        elif label.numpy() == 2:
            labels[2] += 1
            if sizeOfEach > labels[2]:
                varied_dataset_image.append(image)
                varied_dataset_label.append(label)
                two_indices.append(count)
        elif label.numpy() == 3:
            labels[3] += 1
            if sizeOfEach > labels[3]:
                varied_dataset_image.append(image)
                varied_dataset_label.append(label)
                three_indices.append(count)
        elif label.numpy() == 4:
            labels[4] += 1
            if sizeOfEach > labels[4]:
                varied_dataset_image.append(image)
                varied_dataset_label.append(label)
                four_indices.append(count)
        count += 1
    if return_data_or_amount:
        return tf.data.Dataset.from_tensor_slices((varied_dataset_image, varied_dataset_label))
    else:
        return labels

#Function calculates the amount of the occurances of the label with lowest occurances
def get_lowest_size(dataset):
    label_occurances = dataset_with_same_amount(dataset, 1, False)
    return min(label_occurances[0],label_occurances[1],label_occurances[2],label_occurances[3],label_occurances[4])























#
# TFRECORDS DATA SAVE TRY
#
# def _bytes_feature(value):
#   """Returns a bytes_list from a string / byte."""
#   if isinstance(value, type(tf.constant(0))):
#     value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.ndarray.tolist( value)]))
# def _bytes_feature(value):
#   """Returns a bytes_list from a string / byte."""
#   if isinstance(value, type(tf.constant(0))):
#     value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring]))
# def _int64_feature(value):
#   """Returns an int64_list from a bool / enum / int / uint."""
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def _floats_feature(value):
#   nval = (value.numpy()).ravel()
#   return tf.train.Feature(float_list=tf.train.FloatList(value=nval))
#
# def serialize_example(feature0, feature1):
#   """
#   Creates a tf.Example message ready to be written to a file.
#   """
#   # Create a dictionary mapping the feature name to the tf.Example-compatible
#   # data type.
#   feature = {
#       'image': _floats_feature(feature0),
#       'label': _int64_feature(feature1)
#   }
#   example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#   return example_proto.SerializeToString()
#
#
# with tf.io.TFRecordWriter('test.tfrecords') as writer:
#   for x,l in sample_dataset:
#     example = serialize_example(x,l)
#     writer.write(example)
#
# filenames = ['test.tfrecords']
# raw_dataset = tf.data.TFRecordDataset(filenames)

# def decode1(serialized_example):
#     feature_description = {
#         'image': tf.io.FixedLenFeature([], tf.train.BytesList, default_value=''),
#         'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
#     }
#     feature = tf.io.parse_single_example(serialized_example, feature_description)
#
#     # 2. Convert the data
#     image = tf.io.decode_raw(feature['image'], tf.uint8)
#     label = feature['label']
#     # 3. reshape
#     image = tf.reshape(image, [-1, 299, 299,1])
#     return image, label



#raw_dataset.map(decode)
# print(raw_dataset)
# for raw_record in raw_dataset.take(1):
#     print(raw_record)
#     example = tf.train.Example()
#     print(example.ParseFromString(raw_record.numpy()))
#

