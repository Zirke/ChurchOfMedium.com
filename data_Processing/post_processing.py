import matplotlib.pyplot as plt
import tensorflow as tf

# Multi-label names
class_names = ['Negative', 'Benign calcification', 'Benign mass', 'Malignant calcification', 'Malignant mass']
binary_names = ['Negative', 'Positive']

# print and plot history
def plot_history(history):
    print('\nhistory dict:', history.history)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.1, 1])
    plt.legend(loc='lower right')

    plt.show()


def plot_multi_label_predictions(batched_testing_data, model, n):
    # Make predictions for images in testing dataset
    for image, label in batched_testing_data.take(n):
        predictions = model.predict(image.numpy())
        converted_image = tf.reshape(image.numpy()[0], [299, 299])
        plt.imshow(converted_image, cmap=plt.cm.binary)
        plt.xlabel('True Value: %s,\n Predicted Values:'
                   '\nNegative:                [%0.2f], '
                   '\nBenign Calcification:    [%0.2f]'
                   '\nBenign Mass:             [%0.2f]'
                   '\nMalignant Calcification: [%0.2f]'
                   '\nMalignant Mass:          [%0.2f]' % (class_names[label.numpy()[0]],
                                                           predictions[0, 0],
                                                           predictions[0, 1],
                                                           predictions[0, 2],
                                                           predictions[0, 3],
                                                           predictions[0, 4]
                                                           ))
        plt.show()


def plot_binary_label_predictions(batched_testing_data, model, n):
    for image, label in batched_testing_data.take(n):
        predictions = model.predict(image.numpy())

        converted_image = tf.reshape(image.numpy()[0], [299, 299])
        plt.imshow(converted_image, cmap=plt.cm.binary)
        plt.xlabel('True Value: %s,'
                   '\nPredicted Values:'
                   '\nProbability of cancer [%0.2f], ' % (binary_names[label.numpy()[0]], predictions[0, 0] * 100))

        plt.show()
