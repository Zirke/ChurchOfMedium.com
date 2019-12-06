import datetime
import os
import sys
import re
import tensorflow as tf
import numpy as np


# Callback that stops training when it reaches a single
class manual_stopping_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') - logs.get('val_accuracy')) > 0.1:
            print("\nTraining Accuracy is too large compared to validation accuracy")
            self.model.stop_training = True


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
                                                  monitor='val_categorical_accuracy',
                                                  save_best_only=True,
                                                  mode='max',
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
                                                  monitor='val_categorical_accuracy',
                                                  save_best_only=True,
                                                  mode='max',
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  )


class SavePredCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, filepath):
        self.out_log = []
        self.val_data = val_data
        self.file_path = filepath

    def on_train_end(self, logs={}):
        file = open(self.file_path + '.txt', "a")
        file.write('Epoch :\n')
        batch_counter = 0
        for batch, label_batch in self.val_data:
            file.write("batch :" + str(batch_counter) + '\n')
            for image, label in zip(batch, label_batch):
                image = tf.reshape(image.numpy(), [1, 299, 299, 1])
                file.write(str(self.model.predict(image)) + '#' + str(label.numpy()) + '\n')
            batch_counter += 1
        file.close()


def load_saved_pred(filepath):
    prediction, true_label, batches = [], [], []
    with open(filepath, 'r') as myfile:
        data = myfile.read()
        data = data.split('Epoch')
        for epoch in data:
            if not len(epoch.split('batch')) > 1:
                continue
            batches = epoch.split('batch')
            for pred in batches:
                if not len(pred.split('\n')) > 1:
                    continue
                single_pred = pred.split('\n')
                for sp in single_pred:
                    if len(sp.split('#')) > 1 and ':' not in sp:
                        print(sp)
                        value = sp.split('[[')[1].split(']')[0]
                        label = sp.split('#[')[1].split(']')[0]
                        value = re.sub("\s+", ",", value.strip())
                        value = [float(x) for x in value.split(',')]
                        value = softmax_to_binary_value(value)
                        label = re.sub("\s+", ",", label.strip())
                        label = [int(x) for x in label.split(',')]
                        prediction.append(value)
                        true_label.append(label)

            yield prediction, true_label


def confusion_matrix(prediction, true_label, confusion_matrix):
    if true_label == [1, 0, 0, 0, 0]:
        confusion_matrix[0] = v_add(confusion_matrix[0], prediction)
    elif true_label == [0, 1, 0, 0, 0]:
        confusion_matrix[1] = v_add(confusion_matrix[1], prediction)
    elif true_label == [0, 0, 1, 0, 0]:
        confusion_matrix[2] = v_add(confusion_matrix[2], prediction)
    elif true_label == [0, 0, 0, 1, 0]:
        confusion_matrix[3] = v_add(confusion_matrix[3], prediction)
    elif true_label == [0, 0, 0, 0, 1]:
        confusion_matrix[4] = v_add(confusion_matrix[4], prediction)
    else:
        print('Error Matrix Addition')

    return confusion_matrix


def v_add(vec1, vec2):
    return [vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2], vec1[3] + vec2[3], vec1[4] + vec2[4]]


def softmax_to_binary_value(prediction):
    index = 0
    value = prediction[0]
    if value == prediction[1] and value == prediction[2] and value == prediction[3] and value == prediction[4]:
        return [0,0,0,0,0]

    for numbers in range(1, len(prediction)):
        if value < prediction[numbers]:
            value = prediction[numbers]
            index = numbers

    for numbers in range(0, len(prediction)):
        if numbers == index:
            prediction[numbers] = 1
        else:
            prediction[numbers] = 0

    return prediction

prediction_iter = load_saved_pred('C:/Users/120392/Desktop/Training/history/final/c2onfusion_matrix.txt')

for pred, label in prediction_iter:
    confusion_matrix2 = [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]
    for pre, lbl in zip(pred, label):
        confusion_matrix(pre, lbl, confusion_matrix2)
    value_in_matrix = 0
    for row in confusion_matrix2:
        print(row)
        for x in row:
            value_in_matrix += x

    print(value_in_matrix)
