import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import os
from callback import *
from data_preprocess import *
from post_processing import *
from sequential_model import *
from advanced_model import *
from fastcompile import *

# When changes are made to the constants, changes need to be done in sequential_model as well
FILE_SIZE = 7305  # Training dataset size
TEST_SIZE = 500  # Validation and test dataset size
BATCH_SIZE = 32

# Get datasets for training, validation, and testing
parsed_training_data, parsed_val_data, parsed_testing_data = process_dataset()

# batching the dataset into 32-size minibatches
batched_training_data = parsed_training_data.shuffle(buffer_size=1024).batch(BATCH_SIZE)    # BATCH_SIZE
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()                               # BATCH_SIZE
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()                       # BATCH_SIZE


# initializing the callback
callback = myCallback()
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

model = MyModel()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
        batched_training_data,
        steps_per_epoch = FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
        validation_data = batched_val_data,
        validation_steps = TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
        epochs=15,
        verbose=1,  # verbose is the progress bar when training
        callbacks=[callback, cp_callback, tb_callback]
)

# # Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(batched_testing_data, steps= TEST_SIZE // BATCH_SIZE)
print('test loss, test acc:', results)

# History displaying training and validation accuracy
plot_history(history)
plot_multi_label_predictions(batched_testing_data, model, 10)