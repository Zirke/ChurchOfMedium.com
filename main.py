from sorting_hub import *
from callback import *
from data_Processing.post_processing import *
from data_Processing.pre_processing import *
from data_Processing.binary_pre_processing import *
from models import *
import os
import shutil
import webbrowser

import keras_metrics

"""
How to use main.py :

- Enter model;
Five Category is model = Model_Version_1_06c()
Binary Model is model = Model_Version_2_05c()

To change dataset set CONTROL_VARIABLE to one of the following:
 - 'Five'
 - 'Negative'
 - 'BenignC'
 - 'BenignM'
 - 'MalignantC'
 - 'MalignantM'
"""
# TODO add model to CONTROL_VARIABLE
MODEL = Model_Version_2_06f()
CONTROL_VARIABLE = 'MalignantM'

path_holder, type_holder, class_holder = main_management(CONTROL_VARIABLE)
parsed_training_data, parsed_val_data, parsed_testing_data = process_data(path_holder)

FILE_SIZE = len(list(parsed_training_data))  # Training dataset size
TEST_SIZE = len(list(parsed_val_data))  # Validation and test dataset size
BATCH_SIZE = 32
EPOCHS = 10000

# batching the dataset into 32-size mini-batches
batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat(EPOCHS)  # BATCH_SIZE
batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat(EPOCHS)  # BATCH_SIZE
batched_testing_data = parsed_testing_data.batch(BATCH_SIZE)  # BATCH_SIZE

# Clear Tensorboard
if os.path.isdir('logs'):
    shutil.rmtree('logs')

# initializing the callback
es_callback = early_stopping_callback('val_categorical_accuracy', 5)
ms_callback = manual_stopping_callback()
tb_callback = tensorboard_callback("logs", 1)
model_string = str(MODEL).split(".")
cp_callback = checkpoint_callback(str(model_string[len(model_string) - 2]), type_holder, class_holder)

if __name__ == '__main__':
    sub = MODEL
    sub.model().summary()

MODEL.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.metrics.CategoricalAccuracy(),
                       keras_metrics.categorical_false_negative(),
                       keras_metrics.categorical_false_positive(),
#                       keras_metrics.categorical_metric(),
                       keras_metrics.categorical_true_negative(),
                       keras_metrics.categorical_true_positive(),
                       keras_metrics.precision(),
                       keras_metrics.recall()])

history = MODEL.fit(
    batched_training_data,
    steps_per_epoch=FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
    validation_data=batched_testing_data,
    validation_steps=TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
    epochs=EPOCHS,
    shuffle=True,
    verbose=2,
    callbacks=[es_callback, cp_callback]
)

# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = MODEL.evaluate(batched_val_data, steps=TEST_SIZE // BATCH_SIZE)
print('test loss, test acc:', results)

# History displaying training and validation accuracy
if type_holder == 'five':
    plot_multi_label_predictions(batched_testing_data, MODEL, 10)
elif type_holder == 'binary':
    plot_binary_label_predictions(batched_testing_data, MODEL, 10)

plot_history(history)

# Open Tensorboard
webbrowser.open('http://localhost:6006/')
os.system('tensorboard --logdir logs/')
