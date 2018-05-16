import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import inference

tf.set_random_seed(0)
# Read in training data from train.csv
data_train = pd.read_csv('./data/train.csv')
train_feature = data_train.drop(['label'], axis=1)
train_feature = train_feature.values.astype(dtype=np.float32)
train_feature = train_feature.reshape(42000, 28, 28, 1)

labels_list = data_train['label'].tolist()
label_one_hot = tf.one_hot(labels_list, depth=10)  # 将十进制数表示成one_hot
label_one_hot = tf.Session().run(label_one_hot).astype(dtype=np.float64)


# Read in testing data from test.csv
data_test = pd.read_csv('./data/test.csv')
test_feature = data_test.values.astype(dtype=np.float32)
test_feature = test_feature.reshape(28000, 28, 28, 1)

# Display an image read in from the CSV
# testFeatureVectorsConvoFormat values are: [2, 0, 9, 0, 3, 7, ...]
pixels = test_feature[2].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

# Define Tensorflow graph
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

logits = inference.inference(X)
Y = tf.nn.softmax(logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predictions = tf.argmax(Y, 1)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def getBatch(i, size, trainFeatures, trainLabels):
    startIndex = (i * size) % 42000  # 42000 is the size of the train.csv data set
    endIndex = startIndex + size
    batch_X = trainFeatures[startIndex: endIndex]
    batch_Y = trainLabels[startIndex: endIndex]
    return batch_X, batch_Y

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i):
    # training on batches of 100 images with 100 labels
    size = 100
    batch_X, batch_Y = getBatch(i, size, train_feature, label_one_hot)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    # compute training values
    if i % 20 == 0:
        acc, loss = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print(
            str(i) + ": training accuracy:" + str(acc) + " training loss: " + str(loss) + " (lr:" + str(learning_rate) + ")")

    # compute test values
    if i % 100 == 0:
        acc, loss = sess.run([accuracy, cross_entropy],
                        {X: train_feature[-10000:], Y_: label_one_hot[-10000:], pkeep: 1.0})
        print(str(i) + ": ********* test accuracy:" + str(acc) + " test loss: " + str(loss))

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})


for i in range(10000 + 1):
    training_step(i)

acc, loss = sess.run([accuracy, cross_entropy],
                {X: train_feature[-10000:], Y_: label_one_hot[-10000:], pkeep: 1.0})
print("\n ********* test accuracy:" + str(acc) + " test loss: " + str(loss))

# Get predictions on test data
p = sess.run([predictions], {X: test_feature, pkeep: 1.0})

# Write predictions to csv file
results = pd.DataFrame({'ImageId': pd.Series(range(1, len(p[0]) + 1)), 'Label': pd.Series(p[0])})
results.to_csv('results.csv', index=False)












