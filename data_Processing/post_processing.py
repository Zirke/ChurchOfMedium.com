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
                   '\nNegative:                [%0.3f], '
                   '\nBegign Calcification:    [%0.3f]'
                   '\nBenign Mass:             [%0.3f]'
                   '\nMalignant Calcification: [%0.3f]'
                   '\nMalignant Mass:          [%0.3f]' % (get_class_names(label.numpy()),
                                                           predictions[0, 0],
                                                           predictions[0, 1],
                                                           predictions[0, 2],
                                                           predictions[0, 3],
                                                           predictions[0, 4],))
        plt.show()


def plot_binary_label_predictions(batched_testing_data, model, n):
    for image, label in batched_testing_data.take(n):
        predictions = model(image.numpy())
        converted_image = tf.reshape(image.numpy()[0], [299, 299])
        plt.imshow(converted_image, cmap=plt.cm.binary)
        plt.xlabel('True Value: %d ; %d'
                   '\nPredicted Values: %0.3f ; %0.3f, ' % (label.numpy()[0, 0],
                                                            label.numpy()[0, 1],
                                                            predictions.numpy()[0, 0],
                                                            predictions.numpy()[0, 1]))

        plt.show()


def get_class_names(label_name):
    if label_name[0, 0] == 1:
        return 'Negative'
    elif label_name[0, 1] == 1:
        return 'Benign Calcification'
    elif label_name[0, 2] == 1:
        return 'Benign Mass'
    elif label_name[0, 3] == 1:
        return 'Malignant Calcification'
    elif label_name[0, 4] == 1:
        return 'Malignant Mass'
