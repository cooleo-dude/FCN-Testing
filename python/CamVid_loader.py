# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
import cv2
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


root_dir   = "CamVid/"
data_dir   = os.path.join(root_dir, "train")
label_dir  = os.path.join(root_dir, "train_labels")
label_colors_file = os.path.join(root_dir, "class_dict.csv")
val_data_dir = os.path.join(root_dir, "val")
val_label_dir = os.path.join(root_dir, "val_labels")

num_class = 32
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 720, 960
train_h   = int(h * 2 / 3)  # 480
train_w   = int(w * 2 / 3)  # 640
val_h     = int(h/32) * 32  # 704
val_w     = w               # 960

label2color = {}
color2label = {}
label2index = {}
index2label = {}
idx_mat_res = {}

def parse_label(phase):
    # change label to class index
    f = open(label_colors_file, "r")
    data = list(csv.reader(f, delimiter=","))
    f.close()
    idx = 0
    label2color = {}
    color2label = {}
    label2index = {}
    index2label = {}
    idx_mat_res = {}
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
        label = object[0]
        color = tuple((obj1,obj2,obj3))
        label2color[label] = color
        color2label[color] = label
        label2index[label] = idx
        index2label[idx]   = label
        idx+=1
    
    print("Successfully parsed")
    temp = 0

    if phase == "train":
        dir = label_dir
    elif phase == "val":
        dir = val_label_dir
    
    for idx, name in enumerate(os.listdir(dir)):
        print("Parse %s" % (name))
        img = os.path.join(dir, name)
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
        
        idx_mat = idx_mat.astype(np.uint8)
        idx_mat_res[name] = idx_mat
        print("Finish %s" % (name))
        temp +=1
    
    return idx_mat_res
    
class CamVidDataset(Dataset):
    
    #def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5):
    def __init__(self, data, phase, n_class=num_class, crop=True, flip_rate=0.5):
    #def __init__(self, phase, n_class=num_class, crop=True, flip_rate=0.5):
        #self.data      = pd.read_csv(csv_file)
        self.data      = data
        #self.data = parse_label()
        self.means     = means
        self.n_class   = n_class

        self.flip_rate = flip_rate
        self.crop      = crop

        self.phase = phase
        
        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
        elif phase == 'val':
            self.new_h = val_h
            self.new_w = val_w


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #img_name   = self.data.iloc[:, 0]
        img_names  = list(self.data)
        img_name   = img_names[idx]
        if self.phase == "train":
            dir = label_dir
        elif self.phase == "val":
            dir = val_label_dir
 
        img_dir_name = os.path.join(dir, img_name)
        img        = cv2.cvtColor(cv2.imread(img_dir_name),cv2.COLOR_BGR2RGB)
        #label_name = self.data.iloc[:, 1]
        #label      = np.load(label_name[0])
        label      = self.data[img_name]
        
        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample

def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


if __name__ == "__main__":

    #parse_label()
    
    #train_data = CamVidDataset(csv_file=train_file, phase='train')
    train_data = CamVidDataset(parse_label(), phase='train')
    #train_data = CamVidDataset(phase='train')
    
    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())

        # observe 4th batch
        if i == 3:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
