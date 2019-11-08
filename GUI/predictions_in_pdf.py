import os
from data_Processing.image_cropping import *
import PyPDF2 as pdf
from PyPDF2 import PdfFileWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
import cv2
import tensorflow as tf
import datetime
import reportlab as rl
from data_Processing.image_cropping import *
import sys

os.chdir('../')


def pdf_creation(pdf_name):
    """ creates the canvas """
    return canvas.Canvas(pdf_name, pagesize=A4)


def add_image(file_path, c, position_y):
    """ The cropped images use this method """
    image_in = cv2.imread(file_path)
    size = image_in.shape[:2]
    c.drawImage(file_path, 100, position_y, width=size[0] // 2, height=size[1] // 2, mask='auto')


def initial_image(file_path, c, width, height):
    """ Used for the initial uncropped picture, used on the first page """
    image_in = cv2.imread(file_path)
    size = image_in.shape[:2]
    pic_width = size[0] // 4
    pic_height = size[1] // 4
    c.setFont("Times-Roman", 16)
    c.drawString(width // 2 - 40, height // 2 + pic_height // 2 + 10, 'Original Image:')
    c.drawImage(file_path, width // 2 - pic_width // 2, height // 2 - pic_height // 2, width=pic_width,
                height=pic_height, mask='auto')
    c.setFont("Times-Roman", 12)


def create_header(c, model, file_name):
    """ Creates the header with information of the image, model, and date of prediction"""
    fp = file_name.split('/')
    mdl = model.split('_')
    mdl_vers = classification_type(int(mdl[2]))
    c.setLineWidth(.3)
    c.setFont("Times-Roman", 20)

    c.drawString(30, 753, 'ANALYSIS OF: %s' % fp[2])
    c.line(30, 750, 580, 750)

    c.setFont("Times-Roman", 12)
    c.drawString(445, 725, 'Date:')
    c.drawString(500, 725, str(datetime.datetime.now().date()))
    c.line(480, 723, 580, 723)

    c.drawString(355, 700, 'Model Version:')
    c.drawString(450, 700, model)
    c.line(440, 697, 580, 697)

    c.drawString(30, 672, 'CLASSIFICATION:')
    c.line(140, 669, 580, 669)
    c.drawString(140, 672, ' %s ' % mdl_vers)


def write_predictons(model_version, model_path, img, y):
    predictions = make_prediction(get_model(model_version, model_path), img)
    print(predictions[0, 0])


def classification_type(n):
    if n == 1:
        return 'Five Category'
    if n == 2:
        return 'Binary'
    return 'error'


def get_model(model_version, model_path):
    model = getattr(sys.modules[__name__], model_version)()
    checkpoint_path = model_path + '/cp.ckpt'
    model.load_weights(checkpoint_path)
    return model


def make_prediction(input_model, input_picture):
    img = tf.reshape(input_picture, [-1, 299, 299, 1])
    img = tf.cast(img, tf.float32)
    img /= 255.0
    return input_model.predict(img)


def create_file(file_path, original_file, tr_model_version, model_path):
    file_c = pdf_creation(file_path)
    width, height = A4
    fp = original_file.replace('non_cropped', 'cropped')

    create_header(file_c, model_path, original_file)

    # Initial image
    initial_image(original_file, file_c, width, height)
    file_c.showPage()

    y = height - 200
    next_page = 0
    for filename in os.listdir(fp[:-3]):
        img_fp = fp[:-3] + '/' + filename
        add_image(img_fp, file_c, y)
        write_predictons(tr_model_version, model_path, img_fp, y)
        y -= 200
        next_page += 1
        if next_page % 4 == 0:
            file_c.showPage()
            y = height - 200

    file_c.save()


if __name__ == '__main__':
    image = 'pictures/non_cropped/mdb133.jpg'
    resize_image_padding(image)
    create_file('GUI/withimage.pdf', image, 'trained_five_Models/', 'Model_Version_1_06c')
