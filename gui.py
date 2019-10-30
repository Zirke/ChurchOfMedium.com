import sys

import PyQt5
import numpy as np
from PIL import Image
from PyQt5 import QtGui, QtCore
from PyQt5.Qt import Qt
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, \
    QFileSystemModel, QTreeView, QTextEdit, QListView, QSizePolicy

from models.Model_Version_1_01 import *

# Makes the application scale correct on all resolutions
from models.Model_Version_1_04c import Model_Version_1_04c
from models.Model_Version_1_04f import Model_Version_1_04f
from models.Model_Version_2_69f import Model_Version_2_69f

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

with tf.device('/CPU:0'):
    class App(QWidget):
        def __init__(self):
            super().__init__()
            self.left = 100
            self.top = 100
            self.width = 1050
            self.height = 600
            self.setWindowTitle('Mammogram Prediction')
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
            self.picture_label.adjustSize()

            # Widget for adding prediction text
            self.prediction_text = QTextEdit()
            self.prediction_text.append("Probability of Negative: %s" % prediction[0, 0] )
                                         # "\n\nProbability of benign calcification: %s" % prediction[0, 1] +
                                         # "\n\nProbability of benign mass: %s" % prediction[0, 2] +
                                         # "\n\nProbability of malignant calcification: %s" % prediction[0, 3] +
                                         # "\n\nProbability of malignant mass: %s" % prediction[0, 4])
            self.prediction_text.setReadOnly(True)
            # self.prediction_text.setMaximumSize(400, 299)
            # self.prediction_text.setMinimumSize(400, 299)

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
            self.treeview.setColumnWidth(0, 180)
            self.listview.setRootIndex(self.fileModel.index(path))
            self.treeview.clicked.connect(self.on_treeview_clicked)
            self.listview.clicked.connect(self.on_listview_clicked)

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
            self.hbox_buttom.addWidget(self.picture_label, alignment=Qt.AlignCenter)
            self.hbox_buttom.addWidget(self.prediction_text, alignment=Qt.AlignLeft)
            # self.hbox_buttom.addWidget(self.chartView)
            # self.vbox.addWidget(self.button)

            # p = self.palette()
            # p.setColor(self.backgroundRole(), Qt.white)
            # self.setPalette(p)
            self.sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
            self.setSizePolicy(self.sizePolicy)
            self.show()

        def on_treeview_clicked(self, index):
            path = self.dirModel.fileInfo(index).absoluteFilePath()
            self.listview.setRootIndex(self.fileModel.setRootPath(path))

        def is_png(data):
            return data[:8] == '\x89PNG\x0d\x0a\x1a\x0a'

        def on_listview_clicked(self, index):
            model = self.getModel()
            new_picture = self.fileModel.fileInfo(index).absoluteFilePath()
            try:
                im = Image.open(new_picture)
                new_prediction = self.makePrediction(model, self.convertPictureToNumpy(new_picture))

                self.picture_name_label.setText(new_picture)
                self.picture_label.setPixmap(QtGui.QPixmap(new_picture))
                self.prediction_text.setText("Probability of Negative: %s" % new_prediction[0, 0] +
                                             "\n\nProbability of benign calcification: %s" % new_prediction[0, 1] +
                                             "\n\nProbability of benign mass: %s" % new_prediction[0, 2] +
                                             "\n\nProbability of malignant calcification: %s" % new_prediction[0, 3] +
                                             "\n\nProbability of malignant mass: %s" % new_prediction[0, 4])
            except IOError:
                print('Chosen file is not a picture')

        def getModel(self):
            model = Model_Version_2_69f()
            checkpoint_path = 'trained_Models/model_Version_29-10-2019-H14M58/cp.ckpt.index'
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
