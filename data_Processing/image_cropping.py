import os
import cv2

os.chdir('../')
desired_size = 1196


def make_square():
    image_path = 'pictures/non_cropped/mdb152.jpg'
    image_in = cv2.imread(image_path)
    size = image_in.shape[:2]
    max_dim = max(1196, 1196)
    delta_w = max_dim - size[1]
    delta_h = max_dim - size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    image_out = cv2.copyMakeBorder(image_in, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # cv2.imwrite('pictures/cropped/hej.png', image_out)

    crop_image(image_out)


def crop_image(input_image):
    for r in range(0, input_image.shape[0], 299):
        for c in range(0, input_image.shape[1], 299):
            cv2.imwrite('pictures/cropped/'f"img{r}_{c}.png", input_image[r:r + 299, c:c + 299, :])


make_square()
