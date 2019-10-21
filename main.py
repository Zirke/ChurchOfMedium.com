# from models.sequential_model import *
#from sorting_hub import *
# from sorting_hub import *
from callback import *
from data_Processing.post_processing import *
from data_Processing.pre_processing import *
# from models.sequential_model import *
from models.Model_Version_1_01 import *
from models.Model_version_1_02 import *

"""
Get datasets for training, validation, and testing
process_data(file_path) gives a binary classification dataset, list of all file paths in sorting_hub
process_dataset() gives dataset for 5 classes dataset 

from data_Processing.binary_pre_processing import *
"""

parsed_training_data, parsed_val_data, parsed_testing_data = process_dataset()

FILE_SIZE = len(list(parsed_training_data))                                             # Training dataset size
TEST_SIZE = len(list(parsed_val_data))                                                  # Validation and test dataset size
BATCH_SIZE = 32

# batching the dataset into 32-size mini-batches
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat()  # BATCH_SIZE
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()                             # BATCH_SIZE
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()                     # BATCH_SIZE

# initializing the callback
callback = myCallback()
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

model = Model_Version_1_02()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
        batched_training_data,
    steps_per_epoch=100 // BATCH_SIZE,  # FILE_SIZE
        validation_data = batched_val_data,
        validation_steps = TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
    epochs=1,
        shuffle=True,
        verbose=1,  # verbose is the progress bar when training
        callbacks=[callback, cp_callback, tb_callback]
)

# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(batched_testing_data, steps= TEST_SIZE // BATCH_SIZE)
print('test loss, test acc:', results)

if __name__ == '__main__':
    sub = Model_Version_1_01()
    sub.model().summary()

# History displaying training and validation accuracy
plot_binary_label_predictions(batched_testing_data, model, 10)
plot_history(history)