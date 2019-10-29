from tensorflow import keras

from models.Model_Version_1_04f import Model_Version_1_04f
from models.Model_Version_2_02 import Model_Version_2_02
from models.Model_Version_2_69f import Model_Version_2_69f
from sorting_hub import *
from callback import *
from data_Processing.post_processing import *
from data_Processing.pre_processing import *
from data_Processing.binary_pre_processing import *
from models.Model_Version_1_04a import *
from models.Model_Version_1_05a import *
from models.Model_version_1_03c import *
# from models.Model_Version_2_01c import *


parsed_training_data, parsed_val_data, parsed_testing_data = process_data(negative_bi_file_paths)

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

model = Model_Version_2_69f()

# initializing the callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback(str(model))

if __name__ == '__main__':
    sub = Model_Version_2_69f()
    sub.model().summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

#model.load_weights('trained_Models/model_Version_29-10-2019-H13M50/cp.ckpt')

history = model.fit(
    batched_training_data,
    steps_per_epoch=FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
    validation_data=batched_testing_data,
    validation_steps=85,#TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
    epochs=5,
    shuffle=True,
    verbose=2,  # ,  # verbose is the progress bar when training
    callbacks=[cp_callback]
)

# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
#for image,label in batched_val_data.take(1):
#   model.predict(image.numpy())
results = model.evaluate(batched_val_data, steps=TEST_SIZE // BATCH_SIZE)
print('test loss, test acc:', results)

# History displaying training and validation accuracy
plot_binary_label_predictions(batched_testing_data, model, 10)
# plot_history(history)
