import sys

import numpy as np
from PIL import Image
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QVBoxLayout, QPushButton

from models.Model_Version_1_01 import *


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Mammogram Prediction'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.initUI()

    def initUI(self):
        input_image = self.getFileName()
        prediction = self.makePrediction(self.getModel(), self.convertPictureToNumpy(input_image))

        self.picture_label = QLabel(self)
        picture = QtGui.QPixmap(input_image)
        self.picture_label.setPixmap(picture)

        self.label_negative = QLabel("Probability of Negative: %s" % prediction[0, 0], self)
        self.label_benign_cal = QLabel("Probability of benign calcification: %s" % prediction[0, 1], self)
        self.label_benign_mass = QLabel("Probability of benign mass: %s" % prediction[0, 2], self)
        self.label_malignant_cal = QLabel("Probability of malignant calcification: %s" % prediction[0, 3], self)
        self.label_malignant_mass = QLabel("Probability of malignant mass: %s" % prediction[0, 4], self)

        button = QPushButton('Open New Image', self)
        button.clicked.connect(self.on_click)

        vbox = QVBoxLayout()
        vbox.addWidget(self.picture_label)
        vbox.addWidget(self.label_negative)
        vbox.addWidget(self.label_benign_cal)
        vbox.addWidget(self.label_benign_mass)
        vbox.addWidget(self.label_malignant_cal)
        vbox.addWidget(self.label_malignant_mass)
        vbox.addWidget(button)

        self.setLayout(vbox)
        # self.resize(vbox.width(), vbox.height())
        self.show()

    @pyqtSlot()
    def on_click(self):
        new_picture, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "Vælg Billede",
                                                     "All Files (*);;Python Files (*.py)")
        model = self.getModel()
        new_prediction = self.makePrediction(model, self.convertPictureToNumpy(new_picture))
        self.picture_label.setPixmap(QtGui.QPixmap(new_picture))
        self.label_negative.setText("Probability of negative: %s" % new_prediction[0, 0])
        self.label_benign_cal.setText("Probability of benign calcification: %s" % new_prediction[0, 1])
        self.label_benign_mass.setText("Probability of benign mass: %s" % new_prediction[0, 2])
        self.label_malignant_cal.setText("Probability of malignant calcification: %s" % new_prediction[0, 3])
        self.label_malignant_mass.setText("Probability of malignant mass: %s" % new_prediction[0, 4])

    def getModel(self):
        model = Model_Version_1_01()
        checkpoint_path = "trained_Models/model_Version_21-10-2019-H15M30/cp.ckpt"
        model.load_weights(checkpoint_path)
        return model

    def makePrediction(self, input_model, input_picture):
        image = tf.reshape(input_picture, [-1, 299, 299, 1])
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return input_model.predict(image)

    def getFileName(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)")
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
