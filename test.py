import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(299, 3, activation='relu')
        self.maxpol = MaxPooling2D(2, 2)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(5, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class_names = ['Negative', 'Benign calcification', 'Benign mass', 'Malignant calcification', 'Malignant mass']

# Filepaths
train_path_files = ['training10_0/training10_0.tfrecords',
                    'training10_1/training10_1.tfrecords',
                    'training10_2/training10_2.tfrecords']

val_path_file = ['training10_3/training10_3.tfrecords']

test_path_file = ['training10_4/training10_4.tfrecords']

# Extract data as tfrecord dataset
extracted_train_data = tf.data.TFRecordDataset(train_path_files)
extracted_val_data = tf.data.TFRecordDataset(val_path_file)
extracted_test_data = tf.data.TFRecordDataset(test_path_file)

# Each dataset has a description of the features within it.
feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_normal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.io.FixedLenFeature([], tf.string, default_value='')
}


# Extracting one sample at a time and returning an image and an associated label
def decode(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label']

    image = tf.reshape(image, [299, 299, 1])
    image = tf.cast(image, tf.float32)
    return image, label


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


# 33531 images total
parsed_training_data = extracted_train_data.map(decode)
# 11177 images
parsed_val_data = extracted_val_data.map(decode)
# 11177 images
parsed_testing_data = extracted_test_data.map(decode)
FILE_SIZE = 11177

# batching the dataset into 32-size minibatches
BATCH_SIZE = 32
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat()
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()

# Create an instance of the model
defined_model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(299, 299, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = defined_model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, defined_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, defined_model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 1

for epoch in range(EPOCHS):
    for images, labels in parsed_training_data:
        train_step(images, labels)

    for test_images, test_labels in parsed_testing_data:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
