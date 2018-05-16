# -*- coding: utf-8 -*-
"""
Created on Fri 4 13 17:52:07 2018
@author: yangxiaokui
description:这是一个卷积网络，来实现mnist数据集的训练
"""
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一層卷積層的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二層卷積層的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

#全連接層的節點個數
FC_SIZE = 512

#dropout 防止過擬合
def inference(input_tensor, train=False, regularizer=None):
    #使用命名空間來隔離不同層的變量28x28x1---28x28x32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        #使用邊長爲5的全零填充過濾器
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    #28x28x32 ---14x14x32
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #14x14x32---14x14x64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    #14x14x64---7x7x64
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #pool_shape[0]爲一個batch中數據的個數  把數據拉成向量
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

    #3136---512
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        #只有全連接層權重加入正則化
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        #全連接層使用dropout
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
    #512---10
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit

def loss(logits, labels):
    '''
    :param logits:
    :param labels:
    :return: loss tensor
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + './loss', loss)
    return loss

def backpropagation(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def accuracy(logits, labels):

    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + './accuracy', accuracy)
    return accuracy