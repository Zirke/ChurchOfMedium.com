# from models.sequential_model import *
from sorting_hub import *
# from sorting_hub import *
from callback import *
from data_Processing.post_processing import *
from data_Processing.pre_processing import *
from data_Processing.binary_pre_processing import *
# from models.sequential_model import *
from models.Model_Version_1_01 import *
from models.Model_version_1_02 import *
from models.Model_Version_1_04f import *

"""
Get datasets for training, validation, and testing
process_data(file_path) gives a binary classification dataset, list of all file paths in sorting_hub
process_dataset() gives dataset for 5 classes dataset 

from data_Processing.binary_pre_processing import *
"""

parsed_training_data, parsed_val_data, parsed_testing_data = process_data(five_diagnosis_paths)

FILE_SIZE = len(list(parsed_training_data))  # Training dataset size
TEST_SIZE = len(list(parsed_val_data))  # Validation and test dataset size
BATCH_SIZE = 8

# batching the dataset into 32-size mini-batches
batched_training_data = parsed_training_data.batch(BATCH_SIZE)  # BATCH_SIZE
batched_val_data = parsed_val_data.batch(BATCH_SIZE)  # BATCH_SIZE
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()                     # BATCH_SIZE

# initializing the callback
callback = myCallback()
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

batched_training_data = batched_training_data.shuffle(buffer_size=FILE_SIZE,
                                                      seed=None,
                                                      reshuffle_each_iteration=True).repeat()

batched_val_data = batched_val_data.shuffle(buffer_size=TEST_SIZE,
                                            seed=None,
                                            reshuffle_each_iteration=True).repeat()


model = Model_Version_1_04f()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
        batched_training_data,
        steps_per_epoch = FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
        validation_data = batched_val_data,
        validation_steps = TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
        epochs=400,
        shuffle=True,
        verbose=2,  # verbose is the progress bar when training
)

# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(batched_testing_data, steps=TEST_SIZE // BATCH_SIZE)
print('test loss, test acc:', results)

if __name__ == '__main__':
    sub = Model_Version_1_04f()
    sub.model().summary()

# History displaying training and validation accuracy
plot_multi_label_predictions(batched_testing_data, model, 10)
plot_history(history)