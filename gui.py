import sys

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog

from models.Model_Version_1_01 import *


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Title'
        self.left = 10
        self.top = 10
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, 1000, 1000)

        print(self.makePrediction(self.convertPictureToNumpy(self.openFileNameDialog())))

    def makePrediction(self, input_picture):
        model = Model_Version_1_01()
        checkpoint_path = "trained_Models/model_Version_21-10-2019-H15M30/cp.ckpt"
        model.load_weights(checkpoint_path)
        image = tf.reshape(input_picture, [-1, 299, 299, 1])
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        predictions = model.predict(image)
        return predictions[0, 0]

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            return fileName

    def convertPictureToNumpy(self, filename):
        img = Image.open(filename)
        np_array = np.array(img, dtype='uint8')
        return np_array


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
