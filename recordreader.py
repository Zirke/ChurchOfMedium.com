import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model


# Callback that stops training when it reaches a single
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.93:
            print("\n93% accuracy reached and stopping training for now")
            self.model.stop_training = True


# Multiclass names
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

#batching the dataset into 32-size minibatches
BATCH_SIZE = 32
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat()
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()

#initializing the callback
callback = myCallback()

#define the model for training
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(299, 299, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# verbose is the progress bar when training
history = model.fit(
    batched_training_data,
    steps_per_epoch=1000 // BATCH_SIZE,
    shuffle=True,
    validation_data=batched_val_data,
    validation_steps=300 // BATCH_SIZE,
    epochs=5,
    verbose=1,
    callbacks=[callback]
)

#print and plot history
print('\nhistory dict:', history.history)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

#Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(batched_testing_data, steps=300 // BATCH_SIZE)
print('test loss, test acc:', results)

#Make predictions for images in testing dataset
for image, label in parsed_testing_data.take(50):
    image = tf.reshape(image, [-1, 299, 299, 1])
    predictions = model.predict(image.numpy())
    image = tf.reshape(image, [299, 299])
    plt.imshow(image.numpy(), cmap=plt.cm.binary)
    plt.xlabel('True Value: %s,\n Predicted Values:'
               '\nNegative:                [%0.2f], '
               '\nBenign Calcification:    [%0.2f]'
               '\nBenign Mass:             [%0.2f]'
               '\nMalignant Calcification: [%0.2f]'
               '\nMalignant Mass:          [%0.2f]' % (class_names[label.numpy()],
                                                       predictions[0, 0],
                                                       predictions[0, 1],
                                                       predictions[0, 2],
                                                       predictions[0, 3],
                                                       predictions[0, 4]
                                                       ))
    plt.show()

#######################################################
# class MyModel(Model):
#    def __init__(self):
#        super(MyModel, self).__init__()
#        self.conv1 = Conv2D(299, 3, activation='relu')
#        self.maxpol = MaxPooling2D(2,2)
#        self.flatten = Flatten()
#        self.d1 = Dense(128, activation='relu')
#        self.d2 = Dense(5, activation='softmax')

#    def call(self, x):
#        x = self.conv1(x)
#        x = self.flatten(x)
#        x = self.d1(x)
#        return self.d2(x)


# Create an instance of the model
# defined_model = MyModel()

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# @tf.function
# def train_step(images, labels):
#    with tf.GradientTape() as tape:
#        predictions = defined_model(images)
#        loss = loss_object(labels, predictions)
#    gradients = tape.gradient(loss, defined_model.trainable_variables)
#    optimizer.apply_gradients(zip(gradients, defined_model.trainable_variables))

#    train_loss(loss)
#    train_accuracy(labels, predictions)


# @tf.function
# def test_step(images, labels):
#    predictions = model(images)
#    t_loss = loss_object(labels, predictions)

#   test_loss(t_loss)
#   test_accuracy(labels, predictions)


# EPOCHS = 1

# for epoch in range(EPOCHS):
#    for images, labels in parsed_training_data:
#        train_step(images, labels)

#    for test_images, test_labels in parsed_testing_data:
#        test_step(test_images, test_labels)

#    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#    print(template.format(epoch + 1,
#                          train_loss.result(),
#                          train_accuracy.result() * 100,
#                          test_loss.result(),
#                          test_accuracy.result() * 100))

# Reset the metrics for the next epoch
#    train_loss.reset_states()
#    train_accuracy.reset_states()
#    test_loss.reset_states()
#    test_accuracy.reset_states()
