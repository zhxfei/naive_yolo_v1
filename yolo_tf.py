"""
   File Name   :   yolo_tf.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2022/10/22
   Description :
"""
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from output import YoloOutput
from loss import yolo_loss
from utils import train_data_dir, ProcessGenerator
from lr_sche import CustomLearningRateScheduler, lr_schedule

#
#
lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
nb_boxes = 1
grid_w = 7
grid_h = 7
cell_w = 64
cell_h = 64
img_w = grid_w * cell_w
img_h = grid_h * cell_h


def generate_model():
    model = keras.models.Sequential()
    model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), input_shape=(img_h, img_w, 3), padding='same',
                     activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(1470, activation='sigmoid'))
    model.add(YoloOutput(target_shape=(7, 7, 30)))
    model.summary()
    return model


def train():
    train_datasets = []
    val_datasets = []
    with open(train_data_dir + "/VOCdevkit/2007_train.txt", 'r') as f:
        train_datasets = train_datasets + f.readlines()
    with open(train_data_dir + "/VOCdevkit/2007_val.txt", 'r') as f:
        val_datasets = val_datasets + f.readlines()
    X_train, Y_train, X_val, Y_val = list(), list(), list(), list()
    for item in train_datasets:
        item = item.replace("\n", "").split(" ")
        X_train.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_train.append(arr)

    for item in val_datasets:
        item = item.replace("\n", "").split(" ")
        X_val.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_val.append(arr)

    batch_size = 4
    my_training_batch_generator = ProcessGenerator(X_train, Y_train, batch_size)
    my_validation_batch_generator = ProcessGenerator(X_val, Y_val, batch_size)

    model = generate_model()
    mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    model.compile(loss=yolo_loss, optimizer='adam')
    model.fit(x=my_training_batch_generator,
              steps_per_epoch=int(len(X_train) // batch_size),
              epochs=135,
              verbose=1,
              workers=4,
              validation_data=my_validation_batch_generator,
              validation_steps=int(len(X_val) // batch_size),
              callbacks=[
                  CustomLearningRateScheduler(lr_schedule),
                  mcp_save
              ])


if __name__ == '__main__':
    train()
