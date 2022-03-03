import time
import os
import cv2
import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import argparse

target_image_path = "img/IanProfile_nb.jpg"

style_reference_image_path = "img/Selfi_paint.jpg"
result_prefix = "face_nb-source"


class VGG_styletransfer:
    def __init__(self):
        width, height = load_img(target_image_path).size
        self.img_height = 400
        self.img_width = int(width * self.img_height / height)

        target_image = K.constant(self.preprocess_image(target_image_path))
        style_reference_image = K.constant(
            self.preprocess_image(style_reference_image_path)
        )
        combination_image = K.placeholder((1, self.img_height, self.img_width, 3))

        input_tensor = K.concatenate(
            [target_image, style_reference_image, combination_image], axis=0
        )
        self.model = vgg19.VGG19(
            input_tensor=input_tensor, weights="imagenet", include_top=False
        )

        total_variation_weight = 1e-4
        style_weight = 1.0
        content_weight = 0.025


class Stickerizer:
    def process(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        mask_edges = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        img_color = cv2.bilateralFilter(image, d=9, sigmaColor=10, sigmaSpace=10)
        img_sticker = cv2.bitwise_and(img_color, img_color, mask=mask_edges)
        return img_sticker


def run_stickerizer(filename):
    name, ext = filename.split(".")
    print("File: " + filename)
    if ext in ["jpg", "png", "PNG"]:
        image = cv2.imread("./img/" + filename)
        stickerizer = Stickerizer()
        image_out = stickerizer.process(image)
        cv2.imwrite("./output/" + name + "_output2.png", image_out)


if __name__ == "__main__":
    for file in os.listdir("./img"):
        run_stickerizer(file)
