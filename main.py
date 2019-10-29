from tensorflow import keras

from models.Model_Version_1_04f import Model_Version_1_04f
from sorting_hub import *
from callback import *
from data_Processing.post_processing import *
from data_Processing.pre_processing import *
from data_Processing.binary_pre_processing import *
from models.Model_Version_1_04a import *
from models.Model_Version_1_05a import *
from models.Model_version_1_03c import *
# from models.Model_Version_2_01c import *


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

# initializing the callback
callback = early_stopping_callback()
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

model = Model_Version_1_04f()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

if __name__ == '__main__':
    sub = Model_Version_1_04f()
    sub.model().summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    batched_training_data,
    steps_per_epoch=FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
    validation_data=batched_testing_data,
    validation_steps=85,#TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
    epochs=50,
    shuffle=True,
    verbose=2  # ,  # verbose is the progress bar when training
    # callbacks=[callback, cp_callback, tb_callback]
)

# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(batched_val_data, steps=TEST_SIZE // BATCH_SIZE)
print('test loss, test acc:', results)

# History displaying training and validation accuracy
plot_multi_label_predictions(batched_testing_data, model, 10)
plot_history(history)
