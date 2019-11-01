import tensorflow as tf
import PyQt5
import numpy as np
from PIL import Image
from PyQt5 import QtGui, QtCore
from PyQt5.Qt import Qt
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, \
    QFileSystemModel, QTreeView, QTextEdit, QListView, QSizePolicy

from models import *

# Makes the application scale correct on all resolutions
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

currently_selected_model = Model_Version_1_04c()
currently_selected_picture = 'Currently No Image Selected'

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
            # input_image = self.getFileName()
            # global currently_selected_picture
            # currently_selected_picture = input_image

            # Widget for showing picture. The QLabel gets a Pixmap added to it
            self.picture_name_label = QLabel(currently_selected_picture)
            self.picture_label = QLabel()
            self.prediction_text = QTextEdit()
            self.prediction_text.setReadOnly(True)

            if currently_selected_picture == 'Currently No Image Selected':
                self.picture = QtGui.QPixmap('GUI/no_image_selected.png')
                self.prediction_text.append('')
                self.prediction_text.setDisabled(True)
            else:
                model = Model_Version_1_04c()
                checkpoint_path = 'trained_Models/Model_Version_1_04c_30-10-2019-H20M16/cp.ckpt'
                model.load_weights(checkpoint_path)
                prediction = self.makePrediction(model, self.convertPictureToNumpy(currently_selected_picture))

                self.picture = QtGui.QPixmap(currently_selected_picture)

                self.prediction_text.append("Probability of Negative: %s" % prediction[0, 0] +
                                            "\n\nProbability of benign calcification: %s" % prediction[0, 1] +
                                            "\n\nProbability of benign mass: %s" % prediction[0, 2] +
                                            "\n\nProbability of malignant calcification: %s" % prediction[0, 3] +
                                            "\n\nProbability of malignant mass: %s" % prediction[0, 4])

            self.resized_picture = self.picture.scaled(299, 299, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.picture_label.setPixmap(self.resized_picture)

            # Tree and List view for file directory overview of pictures
            self.picture_directory_label = QLabel('Select a Picture:')
            picture_dir_path = '\pictures'
            picture_file_path = 'pictures\\'
            self.treeview_picture = QTreeView()
            self.listview_picture = QListView()

            self.dirModel_picture = QFileSystemModel()
            self.dirModel_picture.setRootPath(picture_dir_path)
            self.dirModel_picture.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

            self.fileModel_picture = QFileSystemModel()
            self.fileModel_picture.setRootPath(picture_file_path)
            self.fileModel_picture.setFilter(QDir.NoDotAndDotDot | QDir.Files)

            self.treeview_picture.setModel(self.dirModel_picture)
            self.listview_picture.setModel(self.fileModel_picture)

            self.treeview_picture.setRootIndex(self.dirModel_picture.index(picture_dir_path))
            self.listview_picture.setRootIndex(self.fileModel_picture.index(picture_file_path))
            self.treeview_picture.setCurrentIndex(self.dirModel_picture.index(0, 0))

            self.treeview_picture.clicked.connect(self.on_picture_treeview_clicked)
            self.listview_picture.clicked.connect(self.on_picture_listview_clicked)
            self.treeview_picture.setColumnHidden(1, True)
            self.treeview_picture.setColumnWidth(0, 275)

            # Tree and List view for file directory overview of models
            self.model_directory_label = QLabel('Select a Model:')
            model_path = 'trained_Models\\'
            self.listview_model = QListView()
            self.dirModel_model = QFileSystemModel()

            self.dirModel_model.setRootPath(model_path)
            self.dirModel_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

            self.listview_model.setModel(self.dirModel_model)
            self.listview_model.setRootIndex(self.dirModel_model.index(model_path))
            self.listview_model.clicked.connect(self.on_model_listview_clicked)

            # Layout handling.
            self.vbox = QVBoxLayout()
            self.vbox_left = QVBoxLayout()
            self.vbox_right = QVBoxLayout()
            self.hbox_outer = QHBoxLayout()
            self.hbox_inner = QHBoxLayout()

            self.vbox.addLayout(self.hbox_outer)
            self.hbox_outer.addLayout(self.vbox_left)
            self.hbox_outer.addLayout(self.vbox_right)
            self.vbox_left.addWidget(self.model_directory_label)
            self.vbox_left.addWidget(self.listview_model)
            self.vbox_left.addWidget(self.picture_directory_label)
            self.vbox_left.addLayout(self.hbox_inner)
            self.hbox_inner.addWidget(self.treeview_picture)
            self.hbox_inner.addWidget(self.listview_picture)

            self.vbox_right.addWidget(self.picture_name_label)
            self.vbox_right.addWidget(self.picture_label)
            self.vbox_right.addWidget(self.prediction_text)

            # p = self.palette()
            # p.setColor(self.backgroundRole(), Qt.white)
            # self.setPalette(p)
            self.setLayout(self.vbox)  # This vbox is the outer layer
            self.sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
            self.setSizePolicy(self.sizePolicy)
            self.show()

        def on_picture_treeview_clicked(self, index):
            path = self.dirModel_picture.fileInfo(index).absoluteFilePath()
            self.listview_picture.setRootIndex(self.fileModel_picture.setRootPath(path))

        def is_png(data):
            return data[:8] == '\x89PNG\x0d\x0a\x1a\x0a'

        def on_picture_listview_clicked(self, index):
            self.prediction_text.setDisabled(False)
            try:
                global currently_selected_picture
                currently_selected_picture = self.fileModel_picture.fileInfo(index).absoluteFilePath()
                Image.open(currently_selected_picture)
                new_prediction = self.makePrediction(currently_selected_model,
                                                     self.convertPictureToNumpy(currently_selected_picture))
                self.picture_name_label.setText(currently_selected_picture)
                self.picture_label.setPixmap(QtGui.QPixmap(currently_selected_picture))
                self.prediction_text.setText("Probability of Negative: %s" % new_prediction[0, 0] +
                                             "\n\nProbability of Benign Calcification: %s" % new_prediction[0, 1] +
                                             "\n\nProbability of Benign Mass: %s" % new_prediction[0, 2] +
                                             "\n\nProbability of Malignant Calcification: %s" % new_prediction[0, 3] +
                                             "\n\nProbability of Malignant Mass: %s" % new_prediction[0, 4])
            except IOError:
                print('Chosen file is not a picture')

        def on_model_listview_clicked(self, index):
            selected_model_path = self.dirModel_model.fileInfo(index).absoluteFilePath()
            selected_model_name = os.path.split(selected_model_path)
            split = selected_model_name[1].split('_')
            selected_model_version = split[0] + '_' + split[1] + '_' + split[2] + '_' + split[3]
            global currently_selected_model
            currently_selected_model = self.getModel(selected_model_version, selected_model_path)

        def getModel(self, model_version, model_path):
            model = getattr(sys.modules[__name__], model_version)()
            checkpoint_path = model_path + '/cp.ckpt'
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
