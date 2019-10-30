from sorting_hub import *
from callback import *
from data_Processing.post_processing import *
from data_Processing.pre_processing import *
from data_Processing.binary_pre_processing import *
from models.Model_Version_2_01c import *
import os
import shutil
import webbrowser

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

parsed_training_data = parsed_training_data.shuffle(buffer_size=FILE_SIZE,
                                                    seed=None,
                                                    reshuffle_each_iteration=True)

parsed_val_data = parsed_val_data.shuffle(buffer_size=TEST_SIZE,
                                          seed=None,
                                          reshuffle_each_iteration=True)

# batching the dataset into 32-size mini-batches
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat()  # BATCH_SIZE
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()  # BATCH_SIZE
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()  # BATCH_SIZE

# Clear Tensorboard
if os.path.isdir('logs'):
    shutil.rmtree('logs')

# initializing the callback
callback = early_stopping_callback()
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

model = Model_Version_2_01c()

if __name__ == '__main__':
    sub = Model_Version_2_01c()
    sub.model().summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    batched_training_data,
    steps_per_epoch=FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
    validation_data=batched_testing_data,
    validation_steps=TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
    epochs=10,
    shuffle=True,
    verbose=2,  # ,  # verbose is the progress bar when training
    callbacks=[cp_callback, tb_callback]
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
