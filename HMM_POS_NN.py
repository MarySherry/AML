import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell



def get_data():

    word2idx = {}
    tag2idx = {}
    word_idx = 1
    tag_idx = 1
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    for line in open('train_pos.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])
      
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        else:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []


    Xtest = []
    Ytest = []
    currentX = []
    currentY = []
    for line in open('test_pos.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag = r
            if word in word2idx:
                currentX.append(word2idx[word])
            else: # Labeling words not in train data as unknown
                currentX.append(word_idx)
            currentY.append(tag2idx[tag])
        else:
            Xtest.append(currentX)
            Ytest.append(currentY)
            currentX = []
            currentY = []


    return Xtrain, Ytrain, Xtest, Ytest, word2idx

def flatten(l):
    return [item for sublist in l for item in sublist]

Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data()
V = len(word2idx) + 2 # vocab size (+1 for unknown, +1 b/c start from 1)
K = len(set(flatten(Ytrain)) | set(flatten(Ytest))) + 1 # num classes

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


# Hyperparameters for training
epochs = 5
learning_rate = 1e-2
mu = 0.99
batch_size = 128
hidden_layer_size = 300
embedding_dim = 150
sequence_length = max(len(x) for x in Xtrain + Xtest)

#Padding sequences using Tensorflow
Xtrain = tf.keras.preprocessing.sequence.pad_sequences(Xtrain, maxlen=sequence_length)
Ytrain = tf.keras.preprocessing.sequence.pad_sequences(Ytrain, maxlen=sequence_length)
Xtest  = tf.keras.preprocessing.sequence.pad_sequences(Xtest,  maxlen=sequence_length)
Ytest  = tf.keras.preprocessing.sequence.pad_sequences(Ytest,  maxlen=sequence_length)
print("Xtrain.shape:", Xtrain.shape)
print("Ytrain.shape:", Ytrain.shape)

#Input params
inputs = tf.placeholder(tf.int32, shape=(None, sequence_length))
targets = tf.placeholder(tf.int32, shape=(None, sequence_length))
num_samples = tf.shape(inputs)[0] # useful for later

# Embedding weights
We = np.random.randn(V, embedding_dim).astype(np.float32)

#Output params
Wo = init_weight(hidden_layer_size, K).astype(np.float32)
bo = np.zeros(K).astype(np.float32)

#Creating tensorflow variables
tfWe = tf.Variable(We)
tfWo = tf.Variable(Wo)
tfbo = tf.Variable(bo)

# Building the RNN unit - Using the GRU RNN
rnn_unit = GRUCell(num_units=hidden_layer_size, activation=tf.nn.relu)


# Outputs from Embedding Layer
x = tf.nn.embedding_lookup(tfWe, inputs)
x = tf.unstack(x, sequence_length, 1)

#Outputs from RNN Layer
outputs, states = get_rnn_output(rnn_unit, x, dtype=tf.float32)
outputs = tf.transpose(outputs, (1, 0, 2))
outputs = tf.reshape(outputs, (sequence_length*num_samples, hidden_layer_size)) # NT x M

# Building the dense layer
logits = tf.matmul(outputs, tfWo) + tfbo # NT x K
predictions = tf.argmax(logits, 1)
predict_op = tf.reshape(predictions, (num_samples, sequence_length))
labels_flat = tf.reshape(targets, [-1])

cost_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_flat))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

# Initialisation
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

#Training Loop
costs = []
n_batches = len(Ytrain) // batch_size
for i in range(epochs):
    n_total = 0
    n_correct = 0

    t0 = datetime.now()
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    cost = 0

    for j in range(n_batches):
        x = Xtrain[j*batch_size:(j+1)*batch_size]
        y = Ytrain[j*batch_size:(j+1)*batch_size]

# Performing the gradient descent
        c, p, _ = sess.run(
            (cost_op, predict_op, train_op),
            feed_dict={inputs: x, targets: y})
        cost += c

#Accuarcy calculations
        for yi, pi in zip(y, p):
            yii = yi[yi > 0]
            pii = pi[yi > 0]
            n_correct += np.sum(yii == pii)
            n_total += len(yii)

        if j % 10 == 0:
            sys.stdout.write(
                "j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
                (j, n_batches, float(n_correct)/n_total, cost)
            )
            sys.stdout.flush()

#Calculating Test Accuracy
    p = sess.run(predict_op, feed_dict={inputs: Xtest, targets: Ytest})
    n_test_correct = 0
    n_test_total = 0
    for yi, pi in zip(Ytest, p):
        yii = yi[yi > 0]
        pii = pi[yi > 0]
        n_test_correct += np.sum(yii == pii)
        n_test_total += len(yii)
    test_acc = float(n_test_correct) / n_test_total

    print(
        "i:", i, "cost:", "%.4f" % cost,
        "train acc:", "%.4f" % (float(n_correct)/n_total),
        "test acc:", "%.4f" % test_acc,
        "time for epoch:", (datetime.now() - t0)
    )
    costs.append(cost)

plt.plot(costs)
plt.show()
