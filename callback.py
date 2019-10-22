import sys
import datetime
import os
import tensorflow as tf

# Callback that stops training when it reaches a single
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.93:
            print("\n98% accuracy reached and stopping training for now")
            self.model.stop_training = True




def tensorboard_callback(log_dir, freq):
    return  tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=freq)

def checkpoint_callback():
    i = datetime.datetime.now().strftime("%d-%m-%Y-H%HM%M")
    if os.path.exists("trained_Models/model_Version_%s" % i):
        print("Model with datetime" + i + " already exists")
        sys.exit()
    else:
        checkpoint_path = "trained_Models/model_Version_" + i + "/cp.ckpt"

    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)