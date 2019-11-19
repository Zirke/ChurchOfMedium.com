import datetime
import os
import sys

import tensorflow as tf


# Callback that stops training when it reaches a single
class manual_stopping_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') - logs.get('val_accuracy')) > 0.1:
            print("\nTraining Accuracy is too large compared to validation accuracy")
            self.model.stop_training = True


"""
class tensorboard_callback(tf.keras.callbacks.TensorBoard):
    def on_train_batch_end(self, batch, logs=None):
"""


def early_stopping_callback(metrics, patience):
    return tf.keras.callbacks.EarlyStopping(monitor=metrics, patience=patience)


def tensorboard_callback(log_dir, freq):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                          histogram_freq=freq,
                                          update_freq='epoch',
                                          write_images=True)


def checkpoint_callback(model, category, *diagnosis):
    if category == 'five':
        i = datetime.datetime.now().strftime("%d-%m-%Y-H%HM%M")
        if os.path.exists("trained_five_Models/%s_%s" % (model, i)):
            print("Model with datetime" + i + " already exists")
            sys.exit()
        else:
            checkpoint_path = "trained_five_Models/" + model + "_" + i + "/cp.ckpt"
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  )

    elif category == 'binary':
        i = datetime.datetime.now().strftime("%d-%m-%Y-H%HM%M")
        if os.path.exists("trained_binary_Models/%s_%s" % (model, i)):
            print("Model with datetime" + i + " already exists")
            sys.exit()
        else:
            checkpoint_path = "trained_binary_Models/" + model + "_" + diagnosis[0] + "_" + i + "/cp.ckpt"
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  )


class SavePredCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, filepath):
        self.out_log = []
        self.val_data = val_data
        self.file_path = filepath

    def on_epoch_end(self, epoch, logs={}):
        file = open(self.file_path, "a")
        file.write('Epoch :' + str(epoch)+'\n')
        batch_counter = 0
        self.out_log.append(epoch)
        for batch, label_batch in self.val_data:
            file.write("batch :" + str(batch_counter) + '\n')
            for image, label in zip(batch, label_batch):
                image = tf.reshape(image.numpy(), [1,299, 299,1])
                file.write(str(self.model.predict(image)) + '#' + str(label.numpy()) + '\n')
            batch_counter += 1
        file.close()