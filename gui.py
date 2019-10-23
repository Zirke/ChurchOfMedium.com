import sys

import PyQt5
import numpy as np
from PIL import Image
from PyQt5 import QtGui, QtCore
from PyQt5.Qt import Qt
from PyQt5.QtChart import QChart, QChartView, QBarCategoryAxis, QBarSet, QBarSeries, QValueAxis
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, \
    QFileSystemModel, QTreeView, QTextEdit, QListView

from models.Model_Version_1_01 import *

# Makes the application scale correct on all resolutions
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Mammogram Prediction'
        self.left = 100
        self.top = 100
        self.width = 1000
        self.height = 600
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.initUI()

    def initUI(self):
        input_image = self.getFileName()
        prediction = self.makePrediction(self.getModel(), self.convertPictureToNumpy(input_image))

        # Widget for showing picture. The QLabel gets a Pixmap added to it
        self.picture_name_label = QLabel(input_image)
        self.picture_label = QLabel(self)
        self.picture = QtGui.QPixmap(input_image)
        self.picture_label.setPixmap(self.picture)

        # Widget for adding prediction text
        self.prediction_text = QTextEdit()
        self.prediction_text.append("Probability of Negative: %s" % prediction[0, 0] +
                                    "\n\nProbability of benign calcification: %s" % prediction[0, 1] +
                                    "\n\nProbability of benign mass: %s" % prediction[0, 2] +
                                    "\n\nProbability of malignant calcification: %s" % prediction[0, 3] +
                                    "\n\nProbability of malignant mass: %s" % prediction[0, 4])
        self.prediction_text.setReadOnly(True)
        # self.prediction_text.setMaximumSize(400, 299)
        # self.prediction_text.setMinimumSize(400, 299)

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

        self.axisX = QBarCategoryAxis()
        self.axisY = QValueAxis()
        self.axisY.setRange(0, 1)
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chartView = QChartView(self.chart)
        self.chartView.setMaximumSize(400, 320)
        self.chartView.setMinimumSize(400, 320)

        # Tree and List view for file directory overview
        path = '/pictures'
        self.treeview = QTreeView()
        self.listview = QListView()

        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(path)
        self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

        self.fileModel = QFileSystemModel()
        self.fileModel.setFilter(QDir.NoDotAndDotDot | QDir.Files)

        self.treeview.setModel(self.dirModel)
        self.listview.setModel(self.fileModel)

        self.treeview.setRootIndex(self.dirModel.index(path))
        self.listview.setRootIndex(self.fileModel.index(path))
        self.treeview.clicked.connect(self.on_clicked)

        # self.directory = QFileSystemModel()
        # self.index = self.directory.index(QDir.currentPath())
        # self.tree = QTreeView(parent=self)
        # self.tree.setModel(self.directory)
        # self.tree.expand(self.index)
        # self.tree.setAnimated(False)
        # self.tree.setIndentation(20)
        # self.tree.setSortingEnabled(True)
        # self.tree.setMaximumSize(500, 300)
        # self.tree.setMinimumSize(500, 300)

        # Button to open new picture file
        # self.button = QPushButton('Open New Image', self)
        # self.button.setMaximumSize(140, 30)
        # self.button.clicked.connect(self.on_clicked)

        # Layout handling.
        self.vbox = QVBoxLayout()
        self.hbox_top = QHBoxLayout()
        self.hbox_buttom = QHBoxLayout()
        self.setLayout(self.vbox)  # This vbox is the outer layer

        self.vbox.addLayout(self.hbox_top)
        self.vbox.addLayout(self.hbox_buttom)

        # Adding widgets to layouts
        self.vbox.addWidget(self.picture_name_label)
        self.hbox_top.addWidget(self.treeview)
        self.hbox_top.addWidget(self.listview)
        self.hbox_buttom.addWidget(self.picture_label, alignment=Qt.AlignLeft)
        self.hbox_buttom.addWidget(self.prediction_text, alignment=Qt.AlignLeft)
        # self.hbox_buttom.addWidget(self.chartView, alignment=Qt.AlignLeft | Qt.AlignCenter)
        # self.vbox.addWidget(self.button)

        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.white)
        # self.setPalette(p)
        self.show()

    def on_clicked(self, index):
        path = self.dirModel.fileInfo(index).absoluteFilePath()
        self.listview.setRootIndex(self.fileModel.setRootPath(path))

    # @pyqtSlot()
    # def on_click(self):
    #     model = self.getModel()
    #     new_picture, _ = self.listview.getOpenFileName(self, "QFileDialog.getOpenFileName()", "Vælg Billede",
    #                                                  "All Files (*);;Python Files (*.py)")
    #     new_prediction = self.makePrediction(model, self.convertPictureToNumpy(new_picture))
    #
    #     self.picture_name_label.setText(new_picture)
    #     self.picture_label.setPixmap(QtGui.QPixmap(new_picture))
    #     self.prediction_text.setText("Probability of Negative: %s" % new_prediction[0, 0] +
    #                                  "\n\nProbability of benign calcification: %s" % new_prediction[0, 1] +
    #                                  "\n\nProbability of benign mass: %s" % new_prediction[0, 2] +
    #                                  "\n\nProbability of malignant calcification: %s" % new_prediction[0, 3] +
    #                                  "\n\nProbability of malignant mass: %s" % new_prediction[0, 4])

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
