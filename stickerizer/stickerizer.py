import time
import os
import cv2
import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import argparse

target_image_path = 'img/IanProfile_nb.jpg'

style_reference_image_path = 'img/Selfi_paint.jpg'
result_prefix = 'face_nb-source'


class VGG_styletransfer:
    def __init__(self):
        width, height = load_img(target_image_path).size
        self.img_height = 400
        self.img_width = int(width * self.img_height / height)

        target_image = K.constant(self.preprocess_image(target_image_path))
        style_reference_image = K.constant(
            self.preprocess_image(style_reference_image_path))
        combination_image = K.placeholder(
            (1, self.img_height, self.img_width, 3))

        input_tensor = K.concatenate(
            [target_image, style_reference_image, combination_image], axis=0)
        self.model = vgg19.VGG19(input_tensor=input_tensor,
                                 weights='imagenet',
                                 include_top=False)

        total_variation_weight = 1e-4
        style_weight = 1.
        content_weight = 0.025

#         output_dict = dict([(layer.name, layer.output)
#                             for layer in self.model.layers])
#         content_layer = 'block5_conv2'
#         style_layers = ['block1_conv1',
#                         'block2_conv1',
#                         'block3_conv1',
#                         'block4_conv1',
#                         'block5_conv1']

#         loss = K.variable(0.)
#         layer_features = output_dict[content_layer]
#         target_image_features = layer_features[0, :, :, :]
#         combination_features = layer_features[2, :, :, :]

#         loss += content_weight * \
#             self.content_loss(target_image_features, combination_features)

#         for layer_name in style_layers:
#             layer_features = output_dict[layer_name]
#             style_reference_features = layer_features[1, :, :, :]
#             combination_features = layer_features[2, :, :, :]
#             sl = self.style_loss(style_reference_features,
#                                  combination_features)
#             loss += (style_weight / len(style_layers)) * sl

#         loss += total_variation_weight * \
#             self.total_variation_loss(combination_image)

#         grads = K.gradients(loss, combination_image)[0]
#         self.fetch_loss_and_grads = K.function(
#             [combination_image], [loss, grads])

#     def preprocess_image(self, image_path):
#         img = load_img(image_path, target_size=(
#             self.img_height, self.img_width))
#         img = img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         img = vgg19.preprocess_input(img)
#         return img

#     def deprocess_image(self, x):
#         x[:, :, 0] += 103.939
#         x[:, :, 1] += 116.779
#         x[:, :, 2] += 123.68
#         x = x[:, :, ::-1]
#         x = np.clip(x, 0, 255).astype('uint8')
#         return x

#     def content_loss(self, base, combination):
#         return K.sum(K.square(combination - base))

#     def gram_matrix(self, x):
#         features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
#         gram = K.dot(features, K.transpose(features))
#         return gram

#     def style_loss(self, style, combination):
#         S = self.gram_matrix(style)
#         C = self.gram_matrix(combination)
#         channels = 3
#         size = self.img_height * self.img_width
#         return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#     def total_variation_loss(self, x):
#         a = K.square(
#             x[:, :self.img_height - 1, :self.img_width - 1, :] -
#             x[:, 1:, :self.img_width - 1, :]
#         )
#         b = K.square(
#             x[:, :self.img_height - 1, :self.img_width - 1, :] -
#             x[:, :self.img_height - 1, 1:, :]
#         )
#         return K.sum(K.pow(a + b, 1.25))


# class Evaluator:
#     def __init__(self, model):
#         self.loss_value = None
#         self.grads_values = None
#         self.model = model

#     def loss(self, x):
#         assert self.loss_value is None
#         x = x.reshape((1, self.model.img_height, self.model.img_width, 3))
#         outs = self.model.fetch_loss_and_grads([x])
#         loss_value = outs[0]
#         grad_values = outs[1].flatten().astype('float64')
#         self.loss_value = loss_value
#         self.grads_values = grad_values
#         return self.loss_value

#     def grads(self, x):
#         assert self.loss_value is not None
#         grad_values = np.copy(self.grads_values)
#         self.loss_value = None
#         self.grads_values = None
#         return grad_values


class Stickerizer():
    def process(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        mask_edges = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        img_color = cv2.bilateralFilter(
            image, d=9, sigmaColor=10, sigmaSpace=10)
        img_sticker = cv2.bitwise_and(img_color, img_color, mask=mask_edges)
        return img_sticker


# def run_vgg_styletransfer():
#     vgg = VGG_styletransfer()
#     evaluator = Evaluator(vgg)

#     iterations = 200

#     x = vgg.preprocess_image(target_image_path)
#     x = x.flatten()
#     for i in range(iterations):
#         print('Start Iteration', i)
#         start_time = time.time()
#         x, min_val, info = fmin_l_bfgs_b(
#             evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
#         print('Current loss value: ', min_val)
#         img = x.copy().reshape((vgg.img_height, vgg.img_width, 3))
#         img = vgg.deprocess_image(img)
#         fname = result_prefix + '_at_iteration_{}.png'.format(i)
#         fpath = os.path.join(
#             '/media/antonio/Data/DataSets/Projects/Stickerizer/Test06-NB-source', fname)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(fpath, img)
#         print('Image saved as: ', fpath)
#         end_time = time.time()
#         print('Iteration {} completed in {} sec,'.format(
#             i, end_time - start_time))


def run_stickerizer(filename):
    name, ext = filename.split('.')
    print('File: ' + filename)
    if ext in ['jpg', 'png', 'PNG']:
        image = cv2.imread('./img/' + filename)
        stickerizer = Stickerizer()
        image_out = stickerizer.process(image)
        cv2.imwrite('./output/' + name + '_output2.png', image_out)


if __name__ == "__main__":
    for file in os.listdir('./img'):
        run_stickerizer(file)
