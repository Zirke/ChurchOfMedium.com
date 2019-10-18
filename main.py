import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import os
from callback import *
from data_preprocess import *
from post_processing import *
from model import *

FILE_SIZE = 7305  # Training dataset size
TEST_SIZE = 500  # Validation and test dataset size
BATCH_SIZE = 32

# Get datasets for training, validation, and testing
parsed_training_data, parsed_val_data, parsed_testing_data = process_dataset()

# batching the dataset into 32-size minibatches
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat()
batched_training_data = batched_training_data.shuffle(buffer_size=1024).batch(BATCH_SIZE)
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()

# initializing the callback
callback = myCallback()
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

# define the model for training
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(299, 299, 1)),
#    tf.keras.layers.MaxPool2D(2, 2),
#    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#    tf.keras.layers.MaxPool2D(2, 2),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.Dropout(0.2),
#    tf.keras.layers.Dense(5, activation='softmax')
#])
model = MyModel()

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

