import tensorflow as tf
import PyQt5
import numpy as np
from PIL import Image
from PyQt5 import QtGui, QtCore
from PyQt5.Qt import Qt
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, \
    QFileSystemModel, QTreeView, QTextEdit, QListView, QSizePolicy, QGridLayout

from models import *

# Makes the application scale correct on all resolutions
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

currently_selected_model = None
currently_selected_model_name = None
currently_selected_picture = 'Currently No Image Selected'

with tf.device('/CPU:0'):
    class App(QWidget):
        def __init__(self):
            super().__init__()
            self.left = 100
            self.top = 100
            self.width = 1100
            self.height = 683
            self.setWindowTitle('Mammogram Prediction')
            self.setGeometry(self.left, self.top, self.width, self.height)

            self.initUI()

        def initUI(self):
            # Widget for showing picture. The QLabel gets a Pixmap added to it to show picture
            self.picture_name_label = QLabel(currently_selected_picture)
            self.picture_label = QLabel()
            self.prediction_text = QTextEdit()
            self.prediction_text.setReadOnly(True)

            self.init_picture_and_predictions()

            self.resized_picture = self.picture.scaled(299, 299, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.picture_label.setPixmap(self.resized_picture)
            self.picture_label.setMinimumWidth(299)
            self.picture_label.setMinimumHeight(299)
            self.picture_label.setContentsMargins(0, 19, 0, 0)

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
            self.treeview_picture.setMinimumWidth(500)
            self.listview_picture.setMinimumWidth(500)

            # List view for file directory overview of five label models
            self.model_directory_label = QLabel('Select a Five Label Model:')
            model_path = 'trained_five_Models\\'
            self.listview_model = QListView()
            self.dirModel_model = QFileSystemModel()

            self.dirModel_model.setRootPath(model_path)
            self.dirModel_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

            self.listview_model.setModel(self.dirModel_model)
            self.listview_model.setRootIndex(self.dirModel_model.index(model_path))
            self.listview_model.clicked.connect(self.on_model_listview_clicked)

            # List view for file directory overview of binary models
            self.model_binary_directory_label = QLabel('Select a Binary Model:')
            model_binary_path = 'trained_binary_Models\\'
            self.listview_model_binary = QListView()
            self.dirModel_model_binary = QFileSystemModel()

            self.dirModel_model_binary.setRootPath(model_binary_path)
            self.dirModel_model_binary.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

            self.listview_model_binary.setModel(self.dirModel_model_binary)
            self.listview_model_binary.setRootIndex(self.dirModel_model_binary.index(model_binary_path))
            self.listview_model_binary.clicked.connect(self.on_model_binary_listview_clicked)

            # Layout handling.

            # self.gridlayout = QGridLayout()
            # self.gridlayout.addWidget(self.model_directory_label, 0, 0)
            # self.gridlayout.setColumnStretch(-15, -11)
            # self.gridlayout.addWidget(self.listview_model, 1, 0)
            # self.gridlayout.addWidget(self.picture_directory_label, 2, 0)
            # self.gridlayout.addWidget(self.treeview_picture, 3, 0)
            #
            # self.gridlayout.addWidget(self.model_binary_directory_label, 0, 1)
            # self.gridlayout.addWidget(self.listview_model_binary, 1, 1)
            # self.gridlayout.addWidget(self.listview_picture, 3, 1)
            #
            # self.gridlayout.addWidget(self.picture_label, 0, 2)
            # self.gridlayout.addWidget(self.picture_name_label, 1, 2)
            # self.gridlayout.addWidget(self.prediction_text, 3, 2)

            self.vbox = QVBoxLayout()
            self.vbox_left = QVBoxLayout()
            self.vbox_right = QVBoxLayout()
            self.hbox_outer = QHBoxLayout()
            self.hbox_top = QHBoxLayout()
            self.hbox_buttom = QHBoxLayout()
            self.vbox_five_label = QVBoxLayout()
            self.vbox_binary = QVBoxLayout()

            self.vbox.addLayout(self.hbox_outer)
            self.hbox_outer.addLayout(self.vbox_left)
            self.hbox_outer.addLayout(self.vbox_right)
            self.vbox_left.addLayout(self.hbox_top)
            self.hbox_top.addLayout(self.vbox_five_label)
            self.hbox_top.addLayout(self.vbox_binary)

            self.vbox_five_label.addWidget(self.model_directory_label)
            self.vbox_five_label.addWidget(self.listview_model)
            self.vbox_binary.addWidget(self.model_binary_directory_label)
            self.vbox_binary.addWidget(self.listview_model_binary)

            self.vbox_left.addWidget(self.picture_directory_label)
            self.vbox_left.addLayout(self.hbox_buttom)
            self.hbox_buttom.addWidget(self.treeview_picture)
            self.hbox_buttom.addWidget(self.listview_picture)

            self.vbox_right.addWidget(self.picture_label, alignment=Qt.AlignHCenter)
            self.vbox_right.addWidget(self.picture_name_label, alignment=Qt.AlignHCenter)
            self.vbox_right.addWidget(self.prediction_text)

            self.vbox_right.setAlignment(Qt.AlignCenter)

            self.setLayout(self.vbox)
            self.sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
            self.setSizePolicy(self.sizePolicy)
            self.show()

        def init_picture_and_predictions(self):
            if currently_selected_picture == 'Currently No Image Selected':
                self.picture = QtGui.QPixmap('GUI/no_image_selected.png')
                self.prediction_text.setText('')

        def on_picture_treeview_clicked(self, index):
            pathof_selected_dir = self.dirModel_picture.fileInfo(index).absoluteFilePath()
            self.listview_picture.setRootIndex(self.fileModel_picture.setRootPath(pathof_selected_dir))

        def on_picture_listview_clicked(self, index):
            global currently_selected_model
            global currently_selected_model_name
            global currently_selected_picture

            currently_selected_picture = self.fileModel_picture.fileInfo(index).absoluteFilePath()
            try:
                Image.open(currently_selected_picture)
                self.picture_name_label.setText(currently_selected_picture)
                self.picture_label.setPixmap(QtGui.QPixmap(currently_selected_picture))
            except IOError:
                print('Exception: Chosen file is not a picture')
            if currently_selected_model is not None:
                new_prediction = self.makePrediction(currently_selected_model,
                                                     self.convertPictureToNumpy(currently_selected_picture))
                split = currently_selected_model_name.split('_')
                if split[4] in ('neg', 'bc', 'bm', 'mc', 'mm'):
                    self.show_binary_prediction(new_prediction, split[4])
                else:
                    self.show_five_prediction(new_prediction)
            else:
                self.prediction_text.setText('No Model is Chosen for Prediction. Choose one to the left.')

        def on_model_listview_clicked(self, index):
            global currently_selected_model
            global currently_selected_model_name
            global currently_selected_picture

            selected_model_path = self.dirModel_model.fileInfo(index).absoluteFilePath()
            selected_model_name = os.path.split(selected_model_path)
            currently_selected_model_name = selected_model_name[1]
            split = selected_model_name[1].split('_')
            selected_model_version = split[0] + '_' + split[1] + '_' + split[2] + '_' + split[3]
            currently_selected_model = self.getModel(selected_model_version, selected_model_path)

            if currently_selected_picture != 'Currently No Image Selected':
                new_prediction = self.makePrediction(currently_selected_model,
                                                     self.convertPictureToNumpy(currently_selected_picture))
                self.picture_name_label.setText(currently_selected_picture)
                self.picture_label.setPixmap(QtGui.QPixmap(currently_selected_picture))
                self.show_five_prediction(new_prediction)

        def on_model_binary_listview_clicked(self, index):
            global currently_selected_model
            global currently_selected_model_name
            global currently_selected_picture

            selected_model_path = self.dirModel_model_binary.fileInfo(index).absoluteFilePath()
            selected_model_name = os.path.split(selected_model_path)
            currently_selected_model_name = selected_model_name[1]
            split = selected_model_name[1].split('_')
            selected_model_category = split[4]
            selected_model_version = split[0] + '_' + split[1] + '_' + split[2] + '_' + split[3]
            currently_selected_model = self.getModel(selected_model_version, selected_model_path)

            if currently_selected_picture != 'Currently No Image Selected':
                new_prediction = self.makePrediction(currently_selected_model,
                                                     self.convertPictureToNumpy(currently_selected_picture))
                self.picture_name_label.setText(currently_selected_picture)
                self.picture_label.setPixmap(QtGui.QPixmap(currently_selected_picture))
                self.show_binary_prediction(new_prediction, selected_model_category)

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

        def show_five_prediction(self, prediction):
            self.prediction_text.setText("Probability of Negative: %s" % prediction[0, 0] +
                                         "\n\nProbability of Benign Calcification: %s" % prediction[0, 1] +
                                         "\n\nProbability of Benign Mass: %s" % prediction[0, 2] +
                                         "\n\nProbability of Malignant Calcification: %s" % prediction[
                                             0, 3] +
                                         "\n\nProbability of Malignant Mass: %s" % prediction[0, 4])

        def show_binary_prediction(self, prediction, category):
            if category == 'neg':
                self.prediction_text.setText("Probability of Negative: %s" % prediction[0, 0])
            elif category == 'bc':
                self.prediction_text.setText("Probability of Benign Calcification: %s" % prediction[0, 0])
            elif category == 'bm':
                self.prediction_text.setText("Probability of Benign Mass: %s" % prediction[0, 0])
            elif category == 'mc':
                self.prediction_text.setText("Probability of Malignant Calcification: %s" % prediction[0, 0])
            elif category == 'mm':
                self.prediction_text.setText("Probability of Malignant Mass: %s" % prediction[0, 0])

        def convertPictureToNumpy(self, filename):
            img = Image.open(filename)
            np_array = np.array(img, dtype='uint8')
            return np_array

        def getFileName(self):
            fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                      "All Files (*);;Python Files (*.py)")
            if fileName:
                return fileName

        def is_png(data):
            return data[:8] == '\x89PNG\x0d\x0a\x1a\x0a'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
