import sys

import numpy as np
from PIL import Image
from PyQt5 import QtGui
from PyQt5.Qt import Qt
from PyQt5.QtChart import QChart, QChartView, QBarCategoryAxis, QBarSet, QBarSeries, QValueAxis
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QVBoxLayout, QPushButton, QHBoxLayout

from models.Model_Version_1_01 import *


# Makes the application scale correct on all resolutions
# PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Mammogram Prediction'
        self.left = 100
        self.top = 100
        self.width = 900
        self.height = 500
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.initUI()

    def initUI(self):
        input_image = self.getFileName()
        prediction = self.makePrediction(self.getModel(), self.convertPictureToNumpy(input_image))

        # Widget for showing picture. The QLabel gets a Pixmap added to it
        self.picture_label = QLabel(self)
        picture = QtGui.QPixmap(input_image)
        self.picture_label.setPixmap(picture)

        # Normal text labels to show predictions
        self.label_negative = QLabel("Probability of Negative: %s" % prediction[0, 0], self)
        self.label_benign_cal = QLabel("Probability of benign calcification: %s" % prediction[0, 1], self)
        self.label_benign_mass = QLabel("Probability of benign mass: %s" % prediction[0, 2], self)
        self.label_malignant_cal = QLabel("Probability of malignant calcification: %s" % prediction[0, 3], self)
        self.label_malignant_mass = QLabel("Probability of malignant mass: %s" % prediction[0, 4], self)

        # QChartView to show a visual representation of probabilities
        self.diag_set_neg = QBarSet('Negative')
        self.diag_set_neg.append(prediction[0, 0])
        self.diag_set_benign_cal = QBarSet('Benign Calcification')
        self.diag_set_benign_cal.append(prediction[0, 1])
        self.diag_set_benign_mass = QBarSet('Benign Mass')
        self.diag_set_benign_mass.append(prediction[0, 2])
        self.diag_set_malignant_cal = QBarSet('Malignant Calcification')
        self.diag_set_malignant_cal.append(prediction[0, 3])
        self.diag_set_malignant_mass = QBarSet('Malignant Mass')
        self.diag_set_malignant_mass.append(prediction[0, 4])

        self.series = QBarSeries()
        self.series.append(self.diag_set_neg)
        self.series.append(self.diag_set_benign_cal)
        self.series.append(self.diag_set_benign_mass)
        self.series.append(self.diag_set_malignant_cal)
        self.series.append(self.diag_set_malignant_mass)

        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.setTitle('Diagnosis Prediction Chart')

        axisX = QBarCategoryAxis()
        axisY = QValueAxis()
        axisY.setRange(0, 1)
        self.chart.addAxis(axisX, Qt.AlignBottom)
        self.chart.addAxis(axisY, Qt.AlignLeft)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chartView = QChartView(self.chart)
        self.chartView.setMaximumSize(600, 320)
        self.chartView.setMinimumSize(600, 320)

        # Button to open new picture file
        button = QPushButton('Open New Image', self)
        button.setMaximumSize(140, 30)
        button.clicked.connect(self.on_click)

        # Layout handling. All widgets gets added to the vbox.
        hbox = QHBoxLayout()
        hbox.addWidget(self.picture_label)
        hbox.addWidget(self.chartView, alignment=Qt.AlignLeft | Qt.AlignCenter)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.label_negative)
        vbox.addWidget(self.label_benign_cal)
        vbox.addWidget(self.label_benign_mass)
        vbox.addWidget(self.label_malignant_cal)
        vbox.addWidget(self.label_malignant_mass)
        vbox.addWidget(button)

        self.setLayout(vbox)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
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
        checkpoint_path = "trained_Models/model_Version_22-10-2019-H09M51/cp.ckpt"
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
