import matplotlib
from PyQt5 import QtWidgets

# from PyQt5.QtWidgets import (Qwidget, QPushButton, QHBoxLayout, QVBoxLayout, QApplication, Qlabel)
matplotlib.use('QT5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from callback import *

from data_Processing.post_processing import *
from fastcompile import *

FILE_SIZE = 7305  # Training dataset size
TEST_SIZE = 500  # Validation and test dataset size
BATCH_SIZE = 32

# Get datasets for training, validation, and testing
# parsed_training_data, parsed_val_data, parsed_testing_data = process_dataset()

# batching the dataset into 32-size minibatches
# batched_training_data = parsed_training_data.batch(BATCH_SIZE).repeat()
# batched_training_data = batched_training_data.shuffle(100).repeat()
# batched_val_data = parsed_val_data.batch(BATCH_SIZE).repeat()
# batched_testing_data = parsed_testing_data.batch(BATCH_SIZE).repeat()

dataset = get_dataset()
dataset = dataset.batch(BATCH_SIZE).repeat()
# initializing the callback
callback = myCallback()
tb_callback = tensorboard_callback("logs", 1)
cp_callback = checkpoint_callback()

# define the model for training
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(299, 299, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    dataset,
    steps_per_epoch=100 // BATCH_SIZE,
    validation_data=dataset,
    validation_steps=100 // BATCH_SIZE,
    epochs=1,
    verbose=1,  # verbose is the progress bar when training
)

# Evaluate the model on unseen testing data
print('\n# Evaluate on test data')
results = model.evaluate(dataset, steps=100 // BATCH_SIZE)
print('test loss, test acc:', results)


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        # uic.loadUi('test.ui', self)

        fig = plot_multi_label_predictions(dataset, model, 1)
        self.plotWidget = FigureCanvasQTAgg(fig)
        lay = QtWidgets.QVBoxLayout(self.content_plot)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)

        # add toolbar
        # self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.plotWidget, self))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())