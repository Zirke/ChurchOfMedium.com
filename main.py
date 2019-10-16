import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.90:
            print("\n90% accurcay reached reached")
            self.model.stop_training = True


train_path_files = ['training10_0/training10_0.tfrecords',
                    'training10_1/training10_1.tfrecords',
                    'training10_2/training10_2.tfrecords',
                    'training10_3/training10_3.tfrecords']

test_path_files = ['training10_4/training10_4.tfrecords']

extracted_data = tf.data.TFRecordDataset(train_path_files)
extracted_test_data = tf.data.TFRecordDataset(test_path_files)

feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_normal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.io.FixedLenFeature([], tf.string, default_value='')
}


def decode(serialized_example):
    feature = tf.io.parse_single_example(serialized_example, feature_description)

    # 2. Convert the data
    image = tf.io.decode_raw(feature['image'], tf.uint8)
    label = feature['label']
    # 3. reshape
    image = tf.reshape(image, [-1, 299, 299, 1])
    return image, label


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


# 44707 images total
parsed_training_data = extracted_data.map(decode)
parsed_testing_data = extracted_test_data.map(decode)

callback = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(299, 299, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# verbose is the progress bar when training
history = model.fit(
    parsed_training_data,
    steps_per_epoch=100,
    shuffle=True,
    validation_data=parsed_testing_data,
    validation_steps=50,
    epochs=2,
    callbacks=[callback]
)

print('\nhistory dict:', history.history)
# model.evaluate(parsed_testing_data)
