# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import csv
import numpy as np
import scipy.misc
import random
import os
import cv2

#############################
    # global variables #
#############################
root_dir          = "CamVid/"
#data_dir          = os.path.join(root_dir, "701_StillsRaw_full")    # train data
data_dir          = os.path.join(root_dir, "train")
#label_dir         = os.path.join(root_dir, "LabeledApproved_full")  # train label
label_dir         = os.path.join(root_dir, "train_labels")
#label_colors_file = os.path.join(root_dir, "label_colors.txt")      # color to label
label_colors_file = os.path.join(root_dir, "class_dict.csv")
val_label_file    = os.path.join(root_dir, "val.csv")               # validation file
train_label_file  = os.path.join(root_dir, "train.csv")             # train file

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
if not os.path.exists(label_idx_dir):
    os.makedirs(label_idx_dir)

label2color = {}
color2label = {}
label2index = {}
index2label = {}


def divide_train_val(val_rate=0.1, shuffle=True, random_seed=None):
    data_list   = os.listdir(data_dir)
    data_len    = len(data_list)
    val_len     = int(data_len * val_rate)

    if random_seed:
        random.seed(random_seed)

    if shuffle:
        data_idx = random.sample(range(data_len), data_len)
    else:
        data_idx = list(range(data_len))

    val_idx     = [data_list[i] for i in data_idx[:val_len]]
    train_idx   = [data_list[i] for i in data_idx[val_len:]]

    # create val.csv
    v = open(val_label_file, "w")
    v.write("img,label\n")
    for idx, name in enumerate(val_idx):
        if 'png' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
        lab_name = lab_name.split(".")[0] + "_L.png.npy"
        v.write("{},{}\n".format(img_name, lab_name))

    # create train.csv
    t = open(train_label_file, "w")
    t.write("img,label\n")
    for idx, name in enumerate(train_idx):
        if 'png' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
        lab_name = lab_name.split(".")[0] + "_L.png.npy"
        t.write("{},{}\n".format(img_name, lab_name))


def parse_label():
    # change label to class index
    #f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
    f = open(label_colors_file, "r")
    data = list(csv.reader(f, delimiter=","))
    f.close()
    #print(data)
    #for idx, line in enumerate(f):
    idx = 0
    for object in data:
        obj1 = object[1]
        obj2 = object[2]
        obj3 = object[3]
        if (object[1]!="r"):
            obj1 = int(object[1])
        if (object[2]!="g"):
            obj2 = int(object[2])
        if (object[3]!="b"):
            obj3 = int(object[3])
        #print(object[3])
        label = object[0]
        #color = tuple([int(x) for x in line.split()[:-1]])
        color = tuple((obj1,obj2,obj3))
        label2color[label] = color
        color2label[color] = label
        label2index[label] = idx
        index2label[idx]   = label
        idx+=1
        # rgb = np.zeros((255, 255, 3), dtype=np.uint8)
        # rgb[..., 0] = color[0]
        # rgb[..., 1] = color[1]
        # rgb[..., 2] = color[2]
        # imshow(rgb, title=label)
    
    print("Successfully parsed")
    
    for idx, name in enumerate(os.listdir(label_dir)):
        filename = os.path.join(label_idx_dir, name)
        if os.path.exists(filename + '.npy'):
            print("Skip %s" % (name))
            continue
        print("Parse %s" % (name))
        img = os.path.join(label_dir, name)
        #img = scipy.misc.imread(img, mode='RGB')
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, weight, _ = img.shape
    
        idx_mat = np.zeros((height, weight))
        for h in range(height):
            for w in range(weight):
                color = tuple(img[h, w])
                try:
                    label = color2label[color]
                    index = label2index[label]
                    idx_mat[h, w] = index
                except:
                    print("error: img:%s, h:%d, w:%d" % (name, h, w))
                #label = color2label[color]
                #index = label2index[label]
                #idx_mat[h, w] = index
        
        idx_mat = idx_mat.astype(np.uint8)
        np.save(filename, idx_mat)
        print("Finish %s" % (name))

    """
    # test some pixels' label    
    #img = os.path.join(label_dir, os.listdir(label_dir)[0])
    img = os.path.join(label_dir, "0001TP_009210_L.png")
    print(img)
    #img = scipy.misc.imread(img, mode='RGB')   
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_cases = [(555, 405), (0, 0), (380, 645), (577, 943)]
    test_ans   = ['Car', 'Building', 'Truck_Bus', 'Car']

    for idx, t in enumerate(test_cases):
        color = img[t]
        print(test_ans[idx])
        print(color)
        print(color2label[tuple(color)])
        assert color2label[tuple(color)] == test_ans[idx]
    """

'''debug function'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
    divide_train_val(random_seed=1)
    parse_label()
