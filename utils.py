"""
   File Name   :   utils.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2022/10/22
   Description :
"""
import argparse
import xml.etree.ElementTree as ET
import os
import cv2 as cv
import numpy as np
from tensorflow import keras

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes_num = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
    'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
    'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}
work_dir = os.getcwd()
data_dir = work_dir + "/data"
train_data_dir = data_dir + '/VOCtrainval_06-Nov-2007'
test_data_dir = data_dir + '/VOCtest_06-Nov-2007'


def notation_process():
    """
    数据标注预处理
    :return:
    """
    for year, image_set in sets:
        base_dir = train_data_dir if image_set in ['val', 'train'] else test_data_dir
        with open(base_dir + '/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set), 'r') as f:
            image_ids = f.read().strip().split()
        with open(base_dir + "/VOCdevkit" + '/%s_%s.txt' % (year, image_set), 'w') as f:
            for image_id in image_ids:
                f.write('%s/%s/VOC%s/JPEGImages/%s.jpg' % (base_dir, "VOCdevkit", year, image_id))
                # convert_annotation(year, image_id, f)
                notation_file = base_dir + '/VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id)
                tree = ET.parse(notation_file)
                root = tree.getroot()
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    classes = list(classes_num.keys())
                    if cls not in classes or int(difficult) == 1:
                        # 过滤掉不在预先定义好的类别里的数据
                        continue
                    cls_id = classes.index(cls)
                    box = obj.find('bndbox')
                    b = (int(box.find('xmin').text),
                         int(box.find('ymin').text),
                         int(box.find('xmax').text),
                         int(box.find('ymax').text))
                    f.write(' ' + ','.join(map(str, b)) + ',' + str(cls_id))
                f.write('\n')


def cv_test():
    """
    读取图片测试
    :return:
    """
    image_path = "./data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000207.jpg"
    label = "1,205,113,320,5"
    l = label.split(',')
    l = np.array(l, dtype=np.int)
    img = cv.imread(image_path)

    # x_min, y_min, x_max, y_max
    box_left_top = (l[0], l[1])
    box_right_bottom = (l[2], l[3])
    point_color = (0, 255, 0)  # BGR
    thickness = 1
    line_type = 4
    cv.rectangle(img, box_left_top, box_right_bottom, point_color, thickness, line_type)
    cv.namedWindow("YOLO V1")
    cv.imshow('YOLO V1', img)
    while True:
        if cv.waitKey(0) == 27:
            break
    cv.destroyAllWindows()


def read_image_with_notation(image_path, labels):
    """
    依据图片的位置和标签信息，生成输出的矩阵
    :param image_path:  "./data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000207.jpg"
    :param labels:  ["1,205,113,320,5", ]
    :return:
    """
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_h, image_w, _ = image.shape
    image = cv.resize(image, (448, 448))

    # normalization
    image = image / 255.0
    # output maxtrix
    label_matrix = np.zeros([7, 7, 30])
    # label is a string linked by notation message
    for label_str in labels:
        label = label_str.split(',')
        _label = np.array(label, dtype=np.int)
        # x_min, y_min, x_max, y_max, cls = _label
        x_min, y_min, x_max, y_max, cls = _label
        x = (x_min + x_max) / 2 / image_w
        y = (y_min + y_max) / 2 / image_h
        w = (x_max - x_min) / image_w
        h = (y_max - y_min) / image_h
        loc = [7 * x, 7 * y]
        # calculate the location of the grid that will make the prediction
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j
        # todo: will fix
        if label_matrix[loc_i, loc_j, 24] == 0:
            label_matrix[loc_i, loc_j, cls] = 1
            label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 24] = 1  # response
    return image, label_matrix


class ProcessGenerator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size):
        #        self.base_dir = train_data_dir if _type in ['val', 'train'] else test_data_dir
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        train_image = []
        train_label = []
        for index in range(0, len(batch_x)):
            img_path = batch_x[index]
            label = batch_y[index]
            image, label_matrix = read_image_with_notation(img_path, label)
            train_image.append(image)
            train_label.append(label_matrix)
        return np.array(train_image), np.array(train_label)


if __name__ == '__main__':

    notation_process()
    train_datasets = []
    val_datasets = []
    with open(train_data_dir + "/VOCdevkit/2007_train.txt", 'r') as f:
        train_datasets = train_datasets + f.readlines()
    with open(train_data_dir + "/VOCdevkit/2007_val.txt", 'r') as f:
        val_datasets = val_datasets + f.readlines()
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    for item in train_datasets:
        item = item.replace("\n", "").split(" ")
        X_train.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_train.append(arr)
    print("X_train sample:")
    print(X_train[0:2])
    print("Y_train sample:")
    print(Y_train[0:2])
    for item in val_datasets:
        item = item.replace("\n", "").split(" ")
        X_val.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_val.append(arr)
    print("X_val sample:")
    print(X_val[0:2])
    print("Y_val sample:")
    print(Y_val[0:2])
    batch_size = 4
    my_training_batch_generator = ProcessGenerator(X_train, Y_train, batch_size )
    my_validation_batch_generator = ProcessGenerator(X_val, Y_val, batch_size)
    x_train, y_train = my_training_batch_generator.__getitem__(0)
    x_val, y_val = my_validation_batch_generator.__getitem__(0)
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
