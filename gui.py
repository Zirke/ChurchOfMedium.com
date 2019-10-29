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
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

with tf.device('/CPU:0'):
    class App(QWidget):
        def __init__(self):
            super().__init__()
            self.left = 100
            self.top = 100
            self.width = 1050
            self.height = 800
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
            self.prediction_text.append("Probability of Negative: %s" % prediction[0, 0] +
                                        "\n\nProbability of benign calcification: %s" % prediction[0, 1] +
                                        "\n\nProbability of benign mass: %s" % prediction[0, 2] +
                                        "\n\nProbability of malignant calcification: %s" % prediction[0, 3] +
                                        "\n\nProbability of malignant mass: %s" % prediction[0, 4])
            self.prediction_text.setReadOnly(True)
            # self.prediction_text.setMaximumSize(400, 299)
            # self.prediction_text.setMinimumSize(400, 299)

            # Tree and List view for file directory overview of pictures
            self.picture_directory_label = QLabel('Select a Picture:')
            picture_path = '\pictures'
            self.treeview_picture = QTreeView()
            self.listview_picture = QListView()

            self.dirModel_picture = QFileSystemModel()
            self.dirModel_picture.setRootPath(picture_path)
            self.dirModel_picture.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

            self.fileModel_picture = QFileSystemModel()
            self.fileModel_picture.setFilter(QDir.NoDotAndDotDot | QDir.Files)

            self.treeview_picture.setModel(self.dirModel_picture)
            self.listview_picture.setModel(self.fileModel_picture)

            self.treeview_picture.setRootIndex(self.dirModel_picture.index(picture_path))
            self.treeview_picture.setColumnWidth(0, 180)
            self.listview_picture.setRootIndex(self.fileModel_picture.index(picture_path))
            self.treeview_picture.clicked.connect(self.on_picture_treeview_clicked)
            self.listview_picture.clicked.connect(self.on_picture_listview_clicked)

            # Tree and List view for file directory overview of models
            self.model_directory_label = QLabel('Select a Model:')
            model_path = '\trained_Models'
            # self.treeview_model = QTreeView()
            self.listview_model = QListView()

            self.dirModel_model = QFileSystemModel()
            self.dirModel_model.setRootPath(model_path)
            self.dirModel_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

            self.fileModel_model = QFileSystemModel()
            self.fileModel_model.setFilter(QDir.NoDotAndDotDot | QDir.Files)

            # self.treeview_model.setModel(self.dirModel_model)
            self.listview_model.setModel(self.fileModel_model)

            # self.treeview_model.setRootIndex(self.dirModel_model.index(model_path))
            # self.treeview_model.setColumnWidth(0, 180)
            self.listview_model.setRootIndex(self.fileModel_model.index(model_path))
            # self.treeview_model.clicked.connect(self.on_model_treeview_clicked)
            self.listview_model.clicked.connect(self.on_model_listview_clicked)

            # Layout handling.
            self.vbox = QVBoxLayout()
            self.hbox_top = QHBoxLayout()
            self.hbox_mid = QHBoxLayout()
            self.hbox_buttom = QHBoxLayout()
            self.setLayout(self.vbox)  # This vbox is the outer layer

            self.vbox.addWidget(self.model_directory_label)
            self.vbox.addLayout(self.hbox_top)
            self.vbox.addWidget(self.picture_directory_label)
            self.vbox.addLayout(self.hbox_mid)
            self.vbox.addLayout(self.hbox_buttom)

            # Adding widgets to layouts
            # self.hbox_top.addWidget(self.treeview_model)
            self.hbox_top.addWidget(self.listview_model)
            self.vbox.addWidget(self.picture_name_label)
            self.hbox_mid.addWidget(self.treeview_picture)
            self.hbox_mid.addWidget(self.listview_picture)

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

        def on_picture_treeview_clicked(self, index):
            path = self.dirModel_picture.fileInfo(index).absoluteFilePath()
            print(path)
            self.listview_picture.setRootIndex(self.fileModel_picture.setRootPath(path))

        def on_model_treeview_clicked(self, index):
            path = self.dirModel_model.fileInfo(index).absoluteFilePath()
            print(path)
            self.listview_model.setRootIndex(self.fileModel_picture.setRootPath(path))

        def is_png(data):
            return data[:8] == '\x89PNG\x0d\x0a\x1a\x0a'

        def on_picture_listview_clicked(self, index):
            model = self.getModel()
            new_picture = self.fileModel_picture.fileInfo(index).absoluteFilePath()
            try:
                Image.open(new_picture)
                new_prediction = self.makePrediction(model, self.convertPictureToNumpy(new_picture))

                self.picture_name_label.setText(new_picture)
                self.picture_label.setPixmap(QtGui.QPixmap(new_picture))
                self.prediction_text.setText("Probability of Negative: %s" % new_prediction[0, 0] +
                                             "\n\nProbability of Benign Calcification: %s" % new_prediction[0, 1] +
                                             "\n\nProbability of Benign Mass: %s" % new_prediction[0, 2] +
                                             "\n\nProbability of Malignant Calcification: %s" % new_prediction[0, 3] +
                                             "\n\nProbability of Malignant Mass: %s" % new_prediction[0, 4])
            except IOError:
                print('Chosen file is not a picture')

        def on_model_listview_clicked(self, index):
            print('Hej')

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
