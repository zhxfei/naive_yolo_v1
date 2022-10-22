"""
   File Name   :   output.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2022/10/22
   Description :
"""
from tensorflow import keras
import keras.backend as K
import tensorflow as tf


class YoloOutput(tf.keras.layers.Layer):
    def __init__(self, target_shape):
        super(YoloOutput, self).__init__()
        self.target_shape = tuple(target_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
        })
        return config

    def call(self, input):
        """

        :param input:
        :return:
        """
        # grids 7x7
        S = [self.target_shape[0], self.target_shape[1]]
        # classes
        C = 20
        # no of bounding boxes per grid
        B = 2
        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B

        # class probabilities
        class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)
        # confidence
        confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)
        # boxes
        boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)
        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs
