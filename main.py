import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import os
from callback import *
from data_preprocess import *
from post_processing import *

FILE_SIZE = 7305  # Training dataset size
TEST_SIZE = 500  # Validation and test dataset size
BATCH_SIZE = 32

# Get datasets for training, validation, and testing
parsed_training_data, parsed_val_data, parsed_testing_data = process_dataset()

# batching the dataset into 32-size minibatches
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat()
batched_training_data = batched_training_data.shuffle(100).repeat()
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()

# initializing the callback
callback = myCallback()

tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

# define the model for training
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

history = model.fit(
    batched_training_data,
    steps_per_epoch=100 // BATCH_SIZE,
    validation_data=batched_val_data,
    validation_steps=100 // BATCH_SIZE,
    epochs=1,
    verbose=1,  # verbose is the progress bar when training
    callbacks=[callback, cp_callback, tb_callback]
)


# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(batched_testing_data, steps=100 // BATCH_SIZE)
print('test loss, test acc:', results)

# History displaying training and validation accuracy
plot_history(history)

plot_multi_label_predictions(batched_testing_data, model, 10)

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
