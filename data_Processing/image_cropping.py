import os
import cv2
from sorting_hub import make_dirs

if __name__ == '__main__':
    os.chdir('../')


def resize_image_padding(image_path):
    image_in = cv2.imread(image_path)
    size = image_in.shape[:2]

    divisor = 0
    while divisor >= 0:
        if divisor * 299 >= size[0]:
            break
        divisor += 1

    max_dim = max(299 * divisor, 299 * divisor)
    delta_w = max_dim - size[1]
    delta_h = max_dim - size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    image_out = cv2.copyMakeBorder(image_in, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    image_name = os.path.split(image_path)[1].split('.')[0]
    crop_image(image_out, image_name)
    return image_name, image_out


def crop_image(input_image, image_name):
    make_dirs('pictures/cropped/%s' % image_name)
    for r in range(0, input_image.shape[0], 299):
        for c in range(0, input_image.shape[1], 299):
            cv2.imwrite('pictures/cropped/' + image_name + '/' + f"%s_{r}_{c}.png" % image_name,
                        input_image[r:r + 299, c:c + 299, :])