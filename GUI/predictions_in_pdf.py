import os
from data_Processing.image_cropping import *
import PyPDF2 as pdf
from PyPDF2 import PdfFileWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
import cv2
from reportlab.lib.colors import Color, black
import tensorflow as tf
from PIL import Image
import datetime
import reportlab as rl
from data_Processing.image_cropping import *
from models import *
import sys
import numpy as np

os.chdir('../')


def pdf_creation(pdf_name):
    """ creates the canvas """
    return canvas.Canvas(pdf_name, pagesize=A4)


def add_image(file_path, c, position_y, width, amount_of_imgs, img_n):
    """ The cropped images use this method """
    image_in = cv2.imread(file_path)
    size = image_in.shape[:2]
    img_width = size[0] // 2
    img_height = size[0] // 2
    c.drawImage(file_path, 100, position_y, width=img_width, height=img_height, mask='auto')
    c.drawString(width - 180, position_y + img_height - 12, "Image: %s of %s" % (img_n + 1, amount_of_imgs))
    c.rect(94, position_y - 6, width - 188, img_height + 12, stroke=1, fill=0)


def initial_image(file_path, c, width, height):
    """ Used for the initial uncropped picture, used on the first page """
    image_in = cv2.imread(file_path)
    size = image_in.shape[:2]
    pic_width = size[0] // 4
    pic_height = size[1] // 4
    c.setFont("Times-Roman", 16)
    c.drawString(width // 2 - 50, height // 2 + pic_height // 2 + 10, 'Original Image:')
    c.drawImage(file_path, width // 2 - pic_width // 2, height // 2 - pic_height // 2, width=pic_width,
                height=pic_height, mask='auto')
    c.setFont("Times-Roman", 12)


def create_header(c, model, file_name, width, height):
    """ Creates the header with information of the image, model, and date of prediction"""
    fp = file_name.split('/')
    mdl = model.split('_')
    mdl_vers = classification_type(int(mdl[2]))
    c.setLineWidth(.3)
    c.setFont("Times-Roman", 28)

    c.line(10, height - 60, width - 20, height - 60)
    c.drawString(width // 2 - 130, height - 135, 'Analysis of %s' % fp[2])
    c.line(10, height - 190, width - 20, height - 190)

    c.rect(width // 4 + 20, 150, 255, 80)

    c.setFont("Times-Roman", 12)
    c.drawString(width // 4 + 20, height // 4 + 30, 'Date:')
    c.drawString(width // 4 + 50, height // 4 + 30, str(datetime.datetime.now().date()))

    c.drawString(width // 2 - 100, height // 4 - 15, 'Model Version:')
    c.drawString(width // 2, height // 4 - 15, model)

    c.drawString(width // 2 - 100, height // 4 - 30, 'Classification:')
    c.drawString(width // 2, height // 4 - 30, mdl_vers)


def write_predictons(model, img, y, c, model_pth):
    x = 300
    op_img = Image.open(img).convert('L')
    np_img = np.array(op_img)
    predictions = make_prediction(model, np_img)
    init_pred = predictions[0, 0]
    index = 0
    for pred in range(1,len(predictions[0])):
        if init_pred < predictions[0, pred]:
            init_pred = predictions[0, pred]
            index = pred

    num_y = y + 110
    for i in range(len(predictions[0])):
        c.drawString(x, num_y, '%0.5f' % predictions[0, i])
        if i == index and i == 0:
            rec_color = Color(0, 100, 0, alpha=0.5)
            c.setFillColor(rec_color)
            c.rect(x - 2, num_y - 5, 49, 20, stroke=1, fill=1)
            c.setFillColor(black)
        elif i == index:
            rec_color = Color(100, 0, 0, alpha=0.5)
            c.setFillColor(rec_color)
            c.rect(x - 2, num_y - 5, 49, 20, stroke=1, fill=1)
            c.setFillColor(black)

        num_y -= 20

    text_x = x + 50
    if (str(model).split('_'))[2] == str(1):
        c.drawString(text_x, y + 110, 'Negative')
        c.drawString(text_x, y + 90, 'Benign calcification')
        c.drawString(text_x, y + 70, 'Benign mass')
        c.drawString(text_x, y + 50, 'Malignant calcification')
        c.drawString(text_x, y + 30, 'Malignant mass')
    elif str(model).split('_')[2] == str(2):
        if str(model_pth).split('_')[6] == 'neg':
            c.drawString(text_x, y + 110, 'Negative')
            c.drawString(text_x, y + 90, 'Positive')
        elif str(model_pth).split('_')[6] == 'bc':
            c.drawString(text_x, y + 110, 'Other diagnosis')
            c.drawString(text_x, y + 90, 'Benign calcification')
        elif str(model_pth).split('_')[6] == 'bm':
            c.drawString(text_x, y + 110, 'Other diagnosis')
            c.drawString(text_x, y + 90, 'Benign mass')
        elif str(model_pth).split('_')[6] == 'mc':
            c.drawString(text_x, y + 110, 'Other diagnosis')
            c.drawString(text_x, y + 90, 'Malignant calcification')
        elif str(model_pth).split('_')[6] == 'mm':
            c.drawString(text_x, y + 110, 'Other diagnosis')
            c.drawString(text_x, y + 90, 'Malignant mass')

    if index > 0:
        return index
    else:
        return 0


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


def padded_image_save(original_file):
    _, padded_image = resize_image_padding(original_file)
    pad_image = Image.fromarray(padded_image)
    make_dirs("pictures/padded_images/")
    pad_image.save("pictures/padded_images/%s" % (original_file.split('/'))[2])


def original_to_padded(original_file):
    return 'pictures/padded_images/%s' % (original_file.split('/'))[2]


def image_product_maker(file_path, c, width, height, pred_array, model):
    """ creates the padded image with marked areas"""
    image_in = cv2.imread(file_path)
    size = image_in.shape[:2]
    pic_width = size[0] // 3
    pic_height = size[1] // 3
    c.setFont("Times-Roman", 16)
    c.drawString(width // 2 - 40, height // 2 + pic_height // 2 + 10, 'Product Image:')
    x = width // 2 - pic_width // 2
    y = height // 2 - pic_height // 2
    c.drawImage(file_path, x, y, width=pic_width, height=pic_height, mask='auto')
    c.setFont("Times-Bold", 14)

    c.rect(x, y - 140, pic_width, 100)
    small_pic_width = pic_width // 4
    small_pic_height = pic_height // 4
    y += pic_height - small_pic_height
    new_x = x
    diagnosis = []
    for i in range(1, len(pred_array)+1):
        if pred_array[i-1] > 0:
            c.setStrokeColorRGB(1, 0, 0)
            c.rect(new_x, y, small_pic_width, small_pic_height, stroke=1, fill=0)
            c.setFillColorRGB(1, 0, 0)
            c.drawString(new_x + 5, y + small_pic_height - 16, "%s" % i)
            c.setFillColorRGB(0, 0, 0)
            c.setStrokeColorRGB(0, 0, 0)
            if pred_array[i-1] not in diagnosis:
                diagnosis.append(pred_array[i-1])
        new_x += pic_width // 4
        if i % 4 == 0:
            new_x = x
            y -= pic_height // 4

    c.setFont("Times-Roman", 12)
    if model == '1':
        end_of_file_text(c, diagnosis, x, y+small_pic_height)


def page_header(c, height, width):
    c.line(30, height - 30, width - 30, height - 30)
    c.drawString(30, height - 27, 'Object Detection')


def end_of_file_text(c, pred_array, x, y):
    c.setFontSize(16)
    c.drawString(x, y - 30, "The Regions of Interest")
    c.setFontSize(12)
    c.drawString(x + 220, y - 60, "Diagnosis of Interest")

    c.rect(x + 220, y - 130, 165, 65)

    for i in range(len(pred_array)):
        if pred_array[i] > 0:
            c.drawString(x + 222, y - 80, " %s " % text_classification(pred_array[i]))
            y -= 13


def text_classification(n):
    if n == 1:
        return '- Benign Calcification'
    elif n == 2:
        return '- Benign Mass'
    elif n == 3:
        return '- Malignant Calcification'
    elif n == 4:
        return '- Malignant Mass'
    else:
        return 'error'


def create_file(file_path, original_file, tr_model_version, model_path):
    file_c = pdf_creation(file_path)
    width, height = A4
    fp = original_file.replace('non_cropped', 'cropped')
    model = get_model(tr_model_version, model_path)
    create_header(file_c, tr_model_version, original_file, width, height)

    # Initial image
    initial_image(original_file, file_c, width, height)
    file_c.drawString(width - 50, 25, str(file_c.getPageNumber()))
    file_c.showPage()
    page_header(file_c, height, width)

    # Prints 16 images and their predictions
    pred_array = []
    y = height - 220
    next_page = 0
    file_c.drawString(width - 50, 25, str(file_c.getPageNumber()))
    amount_of_imgs = len(os.listdir(fp[:-3]))
    for filename in os.listdir(fp[:-3]):
        img_fp = fp[:-3] + '/' + filename
        add_image(img_fp, file_c, y, width, amount_of_imgs, next_page)
        pred_array.append(write_predictons(model, img_fp, y, file_c, model_path))
        y -= 180
        next_page += 1
        if next_page % 4 == 0:
            file_c.showPage()
            page_header(file_c, height, width)
            file_c.drawString(width - 50, 25, str(file_c.getPageNumber()))
            y = height - 220

    padded_image_save(original_file)
    image_product_maker(original_to_padded(original_file), file_c, width, height, pred_array, (tr_model_version.split("_"))[2])
    file_c.save()


if __name__ == '__main__':
    image = 'pictures/non_cropped/mdb136.jpg'
    resize_image_padding(image)
    create_file('GUI/five1.pdf',
                image,
                'Model_Version_1_Final_FD',
                'C:/Users/120392/PycharmProjects/ChurchOfMedium.com/trained_five_Models/Model_Version_Final_FD_05-12-2019-H22M34'
                )
