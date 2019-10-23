import os
import cv2
import glob
import random
import numpy as np
import itertools

def load_dataset():
    dataset = []
    data_list = []
    train_list = []
    train_labels = []
    eval_list = []
    eval_dataset = []
    eval_labels = []
    count = 0
    #train_limit = 4500
    train_limit = 18000

    #for i in range(0, 27):
    for i in range(0, 108):
        for j in range(0, 250):
            d = {}
            d[count] = j
            data_list.append(d)
            count = count + 1
    random.shuffle(data_list)

    count = 0
    for i in range(0, len(data_list)):
        if count < train_limit:
            train_labels.append(data_list[i][[key for key in data_list[i]][0]])
        else:
            eval_labels.append(data_list[i][[key for key in data_list[i]][0]])
        count = count + 1

    count = 0
    for t in range(0, len(data_list)):
    #for t in range(0, 1):
        im = cv2.imread(os.environ.get('DB_PATH')+str([key for key in data_list[t]][0])+'.png', 0)
        nim = cv2.resize(im, (28, 28))
        if count < train_limit:
            dataset.append(nim.tolist())
        else:
            eval_dataset.append(nim.tolist())
        count = count + 1

    dataset = np.array(dataset, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)

    eval_dataset = np.array(eval_dataset, dtype=np.float32)
    eval_labels = np.array(eval_labels, dtype=np.int32)

    dataset = dataset / 255.0
    eval_dataset = eval_dataset / 255.0

    return dataset, train_labels, eval_dataset, eval_labels