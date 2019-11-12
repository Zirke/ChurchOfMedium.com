from sorting_hub import *
from callback import *
from data_Processing.post_processing import *
from data_Processing.pre_processing import *
from data_Processing.binary_pre_processing import *
from models.Model_Version_2_01c import *
from models import *
import os
import shutil
import webbrowser

import keras_metrics

"""
Get datasets for training, validation, and testing
process_data(file_path) gives a binary classification dataset, list of all file paths in sorting_hub
process_dataset() gives dataset for 5 classes dataset

from data_Processing.binary_pre_processing import *
"""
parsed_training_data, parsed_val_data, parsed_testing_data = process_data(five_diagnosis_paths)

FILE_SIZE = len(list(parsed_training_data))  # Training dataset size
TEST_SIZE = len(list(parsed_val_data))  # Validation and test dataset size
BATCH_SIZE = 32
EPOCHS = 50

# batching the dataset into 32-size mini-batches
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat(EPOCHS)  # BATCH_SIZE
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat(EPOCHS)  # BATCH_SIZE
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE)  # BATCH_SIZE
print("here")
# Clear Tensorboard
if os.path.isdir('logs'):
    shutil.rmtree('logs')

model = Model_Version_1_06c()

# initializing the callback
es_callback = early_stopping_callback('val_loss', 5)
ms_callback = manual_stopping_callback()
tb_callback = tensorboard_callback("logs", 1)
model_string = str(model).split(".")
cp_callback = checkpoint_callback(str(model_string[len(model_string) - 2]), 'binary', 'mc')

if __name__ == '__main__':
    sub = model
    sub.model().summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.metrics.CategoricalAccuracy(), keras_metrics.precision(), keras_metrics.recall()])

history = model.fit(
    batched_training_data,
    steps_per_epoch=FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
    validation_data=batched_testing_data,
    validation_steps=TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
    epochs=EPOCHS,
    shuffle=True,
    verbose=2,  # ,  # verbose is the progress bar when training
    callbacks=[cp_callback]
)

# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(batched_val_data, steps=TEST_SIZE // BATCH_SIZE)
print('test loss, test acc:', results)

# History displaying training and validation accuracy
plot_multi_label_predictions(batched_testing_data, model, 10)
plot_history(history)

# Open Tensorboard
webbrowser.open('http://localhost:6006/')
os.system('tensorboard --logdir logs/')


def shuffle(parsed_training_data, parsed_val_data):
    parsed_training_data = parsed_training_data.shuffle(buffer_size=FILE_SIZE,
                                                        seed=None,
                                                        reshuffle_each_iteration=True)

    parsed_val_data = parsed_val_data.shuffle(buffer_size=TEST_SIZE,
                                              seed=None,
                                              reshuffle_each_iteration=True)
