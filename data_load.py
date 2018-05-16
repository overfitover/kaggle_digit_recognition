import numpy as np
import pandas as pd
import time
import csv
import tensorflow as tf
import inference

def loadDataSet(filename):
    '''
    :param filename: 数据文件
    :return:      mat格式的特征和label
    '''
    dataSet = pd.read_csv(filename)
    dataMat = np.mat(dataSet)
    dataLabel = dataMat[:, 0]
    dataMat = dataMat[:,1:]
    m, n = np.shape(dataMat)
    dataMat = np.multiply(dataMat != np.zeros((m, n)), np.ones((m, 1)))
    return dataMat, dataLabel


#print(datamat, datalabel)
#print(np.shape(datamat), np.shape(datalabel))

def saveCsvfile(listfile):
    csvfile = open('KNN_Digit Recognize.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow('ImageId', 'Label')
    data = []
    for i in enumerate(listfile):
        data.append((i[0]+1, i[1]))
    writer.writerows(data)
    csvfile.close()

def get_batch(data, label, batch_size, capacity):

    # 生成队列
    input_queue = tf.train.slice_input_producer([data, label])
    image_contents = input_queue[0]
    label = input_queue[1]
    image_batch, label_batch = tf.train.batch([image_contents, label],
                                              batch_size=batch_size,
                                              num_threads=64,        # 线程
                                              capacity=capacity)     # 队列容量
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch             # (B, 208, 208, 3)  (B)


datamat,datalabel = loadDataSet('./data/train.csv')
a, b = get_batch(datamat, datalabel, 5, 1000)

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#print(a, b)



