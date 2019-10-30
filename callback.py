import datetime
import os
import sys

import tensorflow as tf


# Callback that stops training when it reaches a single
class early_stopping_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') - logs.get('val_accuracy')) > 0.1:
            print("\nTraining Accuracy is too large compared to validation accuracy")
            self.model.stop_training = True


def tensorboard_callback(log_dir, freq):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=freq)


def checkpoint_callback(model):
    i = datetime.datetime.now().strftime("%d-%m-%Y-H%HM%M")
    if os.path.exists("trained_Models/%s_%s" % (model, i)):
        print("Model with datetime" + i + " already exists")
        sys.exit()
    else:
        checkpoint_path = "trained_Models/"+ model + "_" + i + "/cp.ckpt"

    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1,
                                              period=5)
