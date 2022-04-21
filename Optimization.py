"""#5.Design a multilayer neural network, apply the optimizations of learning algorithms.
#5.1.Momentum.
"""

import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
import matplotlib.pyplot as plt

# load datashet
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (60000, 784)) / 255.0
x_test = np.reshape(x_test, (10000, 784)) / 255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.divide(np.matrix(np.exp(x)), np.mat(np.sum(np.exp(x), axis=1)))


def Forwardpass(X, Wh, bh, Wo, bo, Wk, bk):
    zh = X @ Wh.T + bh
    a = sigmoid(zh)
    zk = a @ Wk.T + bk
    b = np.maximum(zk, 0)
    zo = b @ Wo.T + bo
    o = softmax(zo)
    return o


def AccTest(label, prediction):  # calculate the matching score
    OutMaxArg = np.argmax(prediction, axis=1)
    LabelMaxArg = np.argmax(label, axis=1)
    Accuracy = np.mean(OutMaxArg == LabelMaxArg)
    return Accuracy


def ReLU_derivative(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


learningRate = 0.001
Epoch = 50
NumTrainSamples = 60000
NumTestSamples = 10000
NumInputs = 784
NumHiddenUnits = 512
NumClasses = 10
# inital weights
# hidden layer 1
Wh = np.matrix(np.random.uniform(-0.5, 0.5, (512, 784)))
bh = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh = np.zeros((NumHiddenUnits, NumInputs))
Wh_delta = np.zeros((NumHiddenUnits, NumInputs))
dbh = np.zeros((1, NumHiddenUnits))
# hidden layer 2
Wk = np.matrix(np.random.uniform(-0.5, 0.5, (512, 512)))
bk = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWk = np.zeros((NumHiddenUnits, NumHiddenUnits))
Wk_delta = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbk = np.zeros((1, NumHiddenUnits))
# Output layer
Wo = np.random.uniform(-0.5, 0.5, (10, 512))
bo = np.random.uniform(0, 0.5, (1, NumClasses))
dWo = np.zeros((NumClasses, NumHiddenUnits))
Wo_delta = np.zeros((NumClasses, NumHiddenUnits))
dbo = np.zeros((1, NumClasses))
from IPython.display import clear_output

loss = []
Acc = []
Batch_size = 200
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range(0, NumTrainSamples, Batch_size):
        # feed fordware propagation
        Batch_samples = Stochastic_samples[ite:ite + Batch_size]
        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]
        zh = x @ Wh.T + bh
        a = sigmoid(zh)
        zk = a @ Wk.T + bk
        b = np.maximum(zk, 0)
        zo = b @ Wo.T + bo
        o = softmax(zo)
        # calculate cross entropy loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        # calculate back propagation error
        do = o - y
        dk = do @ Wo
        dh = dk @ Wk
        # update weight
        dWo = np.matmul(np.transpose(do), b)
        dks = np.multiply(dk, ReLU_derivative(zk))
        dWk = np.matmul(np.transpose(dks), a)
        dhs = np.multiply(np.multiply(dh, a), 1 - a)
        dWh = np.matmul(np.transpose(dhs), x)
        dbo = np.mean(do)
        dbk = np.mean(dks)
        dbh = np.mean(dhs)
        bo = bo - dbo
        bk = bk - dbk
        bh = bh - dbh
        Wo_delta = 0.95 * Wo_delta + learningRate * dWo / Batch_size
        Wo = Wo - Wo_delta
        Wk_delta = 0.95 * Wk_delta + learningRate * dWk / Batch_size
        Wk = Wk - Wk_delta
        Wh_delta = 0.95 * Wh_delta + learningRate * dWh / Batch_size
        Wh = Wh - Wh_delta
    # Test accuracy with random innitial weights
    prediction = Forwardpass(x_test, Wh, bh, Wo, bo, Wk, bk)
    Acc.append(AccTest(y_test, prediction))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)], Acc, "go")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy test")
    plt.title("Momentum")
    plt.show()
    print('Epoch:', ep)
    print('Accuracy:', AccTest(y_test, prediction))

"""#5.2.ADGRAD."""

import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
import matplotlib.pyplot as plt

# load datashet
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (60000, 784)) / 255.0
x_test = np.reshape(x_test, (10000, 784)) / 255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.divide(np.matrix(np.exp(x)), np.mat(np.sum(np.exp(x), axis=1)))


def Forwardpass(X, Wh, bh, Wo, bo, Wk, bk):
    zh = X @ Wh.T + bh
    a = sigmoid(zh)
    zk = a @ Wk.T + bk
    b = np.maximum(zk, 0)
    zo = b @ Wo.T + bo
    o = softmax(zo)
    return o


def AccTest(label, prediction):  # calculate the matching score
    OutMaxArg = np.argmax(prediction, axis=1)
    LabelMaxArg = np.argmax(label, axis=1)
    Accuracy = np.mean(OutMaxArg == LabelMaxArg)
    return Accuracy


def ReLU_derivative(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


learningRate = 0.5
e = 1e-9
Epoch = 50
NumTrainSamples = 60000
NumTestSamples = 10000
NumInputs = 784
NumHiddenUnits = 512
NumClasses = 10
# inital weights
# hidden layer 1
Wh = np.matrix(np.random.uniform(-0.5, 0.5, (512, 784)))
bh = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh = np.zeros((NumHiddenUnits, NumInputs))
dbh = np.zeros((1, NumHiddenUnits))
at1 = np.zeros((NumHiddenUnits, NumInputs))
# hidden layer 2
Wk = np.matrix(np.random.uniform(-0.5, 0.5, (512, 512)))
bk = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWk = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbk = np.zeros((1, NumHiddenUnits))
at2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
# Output layer
Wo = np.random.uniform(-0.5, 0.5, (10, 512))
bo = np.random.uniform(0, 0.5, (1, NumClasses))
dWo = np.zeros((NumClasses, NumHiddenUnits))
dbo = np.zeros((1, NumClasses))
at = np.zeros((NumClasses, NumHiddenUnits))
from IPython.display import clear_output

loss = []
Acc = []
Batch_size = 200
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range(0, NumTrainSamples, Batch_size):
        # feed fordware propagation
        Batch_samples = Stochastic_samples[ite:ite + Batch_size]
        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]
        zh = x @ Wh.T + bh
        a = sigmoid(zh)
        zk = a @ Wk.T + bk
        b = np.maximum(zk, 0)
        zo = b @ Wo.T + bo
        o = softmax(zo)
        # calculate cross entropy loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        # calculate back propagation error
        do = o - y
        dk = do @ Wo
        dh = dk @ Wk
        # update weight
        dWo = np.matmul(np.transpose(do), b)
        dks = np.multiply(dk, ReLU_derivative(zk))
        dWk = np.matmul(np.transpose(dks), a)
        dhs = np.multiply(np.multiply(dh, a), 1 - a)
        dWh = np.matmul(np.transpose(dhs), x)
        dbo = np.mean(do)
        dbk = np.mean(dks)
        dbh = np.mean(dhs)
        bo = bo - dbo
        bk = bk - dbk
        bh = bh - dbh
        at = at + np.power(dWo, 2)
        n_delta = np.divide(np.multiply(learningRate, dWo / Batch_size), np.sqrt(at + e))
        at2 = at2 + np.power(dWk, 2)
        n_delta2 = np.divide(np.multiply(learningRate, dWk / Batch_size), np.sqrt(at2 + e))
        at1 = at1 + np.power(dWh, 2)
        n_delta1 = np.divide(np.multiply(learningRate, dWh / Batch_size), np.sqrt(at1 + e))
        Wo = Wo - n_delta
        Wk = Wk - n_delta2
        Wh = Wh - n_delta1
    # Test accuracy with random innitial weights
    prediction = Forwardpass(x_test, Wh, bh, Wo, bo, Wk, bk)
    Acc.append(AccTest(y_test, prediction))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)], Acc, "o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy test")
    plt.title("ADAGRAD")
    plt.show()
    print('Epoch:', ep)
    print('Accuracy:', AccTest(y_test, prediction))

"""#5.3.Dropout + Momentum."""

import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
import matplotlib.pyplot as plt

# load datashet
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (60000, 784)) / 255.0
x_test = np.reshape(x_test, (10000, 784)) / 255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")


def dropout(X, drop_prob):
    keep_prob = 1 - drop_ratio
    d = np.random.rand(X.shape[0], X.shape[1]) < keep_prob
    X = np.multiply(X, d)
    X /= keep_prob
    return X


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.divide(np.matrix(np.exp(x)), np.mat(np.sum(np.exp(x), axis=1)))


def Forwardpass(X, Wh, bh, Wo, bo, Wk, bk):
    zh = X @ Wh.T + bh
    a = sigmoid(zh)
    a = dropout(a, drop_ratio)
    zk = a @ Wk.T + bk
    b = np.maximum(zk, 0)
    b = dropout(b, drop_ratio)
    zo = b @ Wo.T + bo
    o = softmax(zo)
    return o


def AccTest(label, prediction):  # calculate the matching score
    OutMaxArg = np.argmax(prediction, axis=1)
    LabelMaxArg = np.argmax(label, axis=1)
    Accuracy = np.mean(OutMaxArg == LabelMaxArg)
    return Accuracy


def ReLU_derivative(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


learningRate = 0.001
Epoch = 50
drop_ratio = 0.1
NumTrainSamples = 60000
NumTestSamples = 10000
NumInputs = 784
NumHiddenUnits = 512
NumClasses = 10
# inital weights
# hidden layer 1
Wh = np.matrix(np.random.uniform(-0.5, 0.5, (512, 784)))
bh = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh = np.zeros((NumHiddenUnits, NumInputs))
Wh_delta = np.zeros((NumHiddenUnits, NumInputs))
dbh = np.zeros((1, NumHiddenUnits))
# hidden layer 2
Wk = np.matrix(np.random.uniform(-0.5, 0.5, (512, 512)))
bk = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWk = np.zeros((NumHiddenUnits, NumHiddenUnits))
Wk_delta = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbk = np.zeros((1, NumHiddenUnits))
# Output layer
Wo = np.random.uniform(-0.5, 0.5, (10, 512))
bo = np.random.uniform(0, 0.5, (1, NumClasses))
dWo = np.zeros((NumClasses, NumHiddenUnits))
Wo_delta = np.zeros((NumClasses, NumHiddenUnits))
dbo = np.zeros((1, NumClasses))
from IPython.display import clear_output

loss = []
Acc = []
Batch_size = 200
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range(0, NumTrainSamples, Batch_size):
        # feed fordware propagation
        Batch_samples = Stochastic_samples[ite:ite + Batch_size]
        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]
        zh = x @ Wh.T + bh
        a = sigmoid(zh)
        a = dropout(a, drop_ratio)
        zk = a @ Wk.T + bk
        b = np.maximum(zk, 0)
        b = dropout(b, drop_ratio)
        zo = b @ Wo.T + bo
        o = softmax(zo)
        # calculate cross entropy loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        # calculate back propagation error
        do = o - y
        dk = do @ Wo
        dh = dk @ Wk
        # update weight
        dWo = np.matmul(np.transpose(do), b)
        dks = np.multiply(dk, ReLU_derivative(zk))
        dWk = np.matmul(np.transpose(dks), a)
        dhs = np.multiply(np.multiply(dh, a), 1 - a)
        dWh = np.matmul(np.transpose(dhs), x)
        dbo = np.mean(do)
        dbk = np.mean(dks)
        dbh = np.mean(dhs)
        bo = bo - dbo
        bk = bk - dbk
        bh = bh - dbh
        Wo_delta = 0.95 * Wo_delta + learningRate * dWo / Batch_size
        Wo = Wo - Wo_delta
        Wk_delta = 0.95 * Wk_delta + learningRate * dWk / Batch_size
        Wk = Wk - Wk_delta
        Wh_delta = 0.95 * Wh_delta + learningRate * dWh / Batch_size
        Wh = Wh - Wh_delta
    # Test accuracy with random innitial weights
    prediction = Forwardpass(x_test, Wh, bh, Wo, bo, Wk, bk)
    Acc.append(AccTest(y_test, prediction))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)], Acc, "ro")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy test")
    plt.title("Dropout + Momentum")
    plt.show()
    print('Epoch:', ep)
    print('Accuracy:', AccTest(y_test, prediction))

"""#5.4.Dropout + ADAGRAD"""

import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
import matplotlib.pyplot as plt

# load datashet
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (60000, 784)) / 255.0
x_test = np.reshape(x_test, (10000, 784)) / 255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")
drop_ratio = 0.1


def dropout(X):
    keep_prob = 1 - drop_ratio
    d = np.random.rand(X.shape[0], X.shape[1]) < keep_prob
    X = np.multiply(X, d)
    X /= keep_prob
    return X


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.divide(np.matrix(np.exp(x)), np.mat(np.sum(np.exp(x), axis=1)))


def Forwardpass(X, Wh, bh, Wo, bo, Wk, bk):
    zh = X @ Wh.T + bh
    a = sigmoid(zh)
    a = dropout(a)
    zk = a @ Wk.T + bk
    b = np.maximum(zk, 0)
    b = dropout(b)
    zo = b @ Wo.T + bo
    o = softmax(zo)
    return o


def AccTest(label, prediction):  # calculate the matching score
    OutMaxArg = np.argmax(prediction, axis=1)
    LabelMaxArg = np.argmax(label, axis=1)
    Accuracy = np.mean(OutMaxArg == LabelMaxArg)
    return Accuracy


def ReLU_derivative(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


learningRate = 0.5
e = 1e-9
Epoch = 50
NumTrainSamples = 60000
NumTestSamples = 10000
NumInputs = 784
NumHiddenUnits = 512
NumClasses = 10
# inital weights
# hidden layer 1
Wh = np.matrix(np.random.uniform(-0.5, 0.5, (512, 784)))
bh = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh = np.zeros((NumHiddenUnits, NumInputs))
dbh = np.zeros((1, NumHiddenUnits))
at1 = np.zeros((NumHiddenUnits, NumInputs))
# hidden layer 2
Wk = np.matrix(np.random.uniform(-0.5, 0.5, (512, 512)))
bk = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWk = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbk = np.zeros((1, NumHiddenUnits))
at2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
# Output layer
Wo = np.random.uniform(-0.5, 0.5, (10, 512))
bo = np.random.uniform(0, 0.5, (1, NumClasses))
dWo = np.zeros((NumClasses, NumHiddenUnits))
dbo = np.zeros((1, NumClasses))
at = np.zeros((NumClasses, NumHiddenUnits))
from IPython.display import clear_output

loss = []
Acc = []
Batch_size = 200
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range(0, NumTrainSamples, Batch_size):
        # feed fordware propagation
        Batch_samples = Stochastic_samples[ite:ite + Batch_size]
        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]
        zh = x @ Wh.T + bh
        a = sigmoid(zh)
        a = dropout(a)
        zk = a @ Wk.T + bk
        b = np.maximum(zk, 0)
        b = dropout(b)
        zo = b @ Wo.T + bo
        o = softmax(zo)
        # calculate cross entropy loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        # calculate back propagation error
        do = o - y
        dk = do @ Wo
        dh = dk @ Wk
        # update weight
        dWo = np.matmul(np.transpose(do), b)
        dks = np.multiply(dk, ReLU_derivative(zk))
        dWk = np.matmul(np.transpose(dks), a)
        dhs = np.multiply(np.multiply(dh, a), 1 - a)
        dWh = np.matmul(np.transpose(dhs), x)
        dbo = np.mean(do)
        dbk = np.mean(dks)
        dbh = np.mean(dhs)
        bo = bo - dbo
        bk = bk - dbk
        bh = bh - dbh
        at = at + np.power(dWo, 2)
        n_delta = np.divide(np.multiply(learningRate, dWo / Batch_size), np.sqrt(at + e))
        at2 = at2 + np.power(dWk, 2)
        n_delta2 = np.divide(np.multiply(learningRate, dWk / Batch_size), np.sqrt(at2 + e))
        at1 = at1 + np.power(dWh, 2)
        n_delta1 = np.divide(np.multiply(learningRate, dWh / Batch_size), np.sqrt(at1 + e))
        Wo = Wo - n_delta
        Wk = Wk - n_delta2
        Wh = Wh - n_delta1
    # Test accuracy with random innitial weights
    prediction = Forwardpass(x_test, Wh, bh, Wo, bo, Wk, bk)
    Acc.append(AccTest(y_test, prediction))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)], Acc, "yo")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy test")
    plt.title("Dropout + ADAGRAD")
    plt.show()
    print('Epoch:', ep)
    print('Accuracy:', AccTest(y_test, prediction))

"""#5.5.Comparison"""

import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
import matplotlib.pyplot as plt

# load datashet
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (60000, 784)) / 255.0
x_test = np.reshape(x_test, (10000, 784)) / 255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.divide(np.matrix(np.exp(x)), np.mat(np.sum(np.exp(x), axis=1)))


def Forwardpass(X, Wh, bh, Wo, bo, Wk, bk):
    zh = X @ Wh.T + bh
    a = sigmoid(zh)
    zk = a @ Wk.T + bk
    b = np.maximum(zk, 0)
    zo = b @ Wo.T + bo
    o = softmax(zo)
    return o


def AccTest(label, prediction):  # calculate the matching score
    OutMaxArg = np.argmax(prediction, axis=1)
    LabelMaxArg = np.argmax(label, axis=1)
    Accuracy = np.mean(OutMaxArg == LabelMaxArg)
    return Accuracy


def AccTest1(label1, prediction1):  # calculate the matching score
    OutMaxArg1 = np.argmax(prediction1, axis=1)
    LabelMaxArg1 = np.argmax(label1, axis=1)
    Accuracy1 = np.mean(OutMaxArg1 == LabelMaxArg1)
    return Accuracy1


def AccTest2(label2, prediction2):  # calculate the matching score
    OutMaxArg2 = np.argmax(prediction2, axis=1)
    LabelMaxArg2 = np.argmax(label2, axis=1)
    Accuracy2 = np.mean(OutMaxArg2 == LabelMaxArg2)
    return Accuracy2


def ReLU_derivative(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


def dropout(X, drop_prob):
    keep_prob = 1 - drop_ratio
    d = np.random.rand(X.shape[0], X.shape[1]) < keep_prob
    X = np.multiply(X, d)
    X /= keep_prob
    return X


def Forwardpass1(X, Wh, bh, Wo, bo, Wk, bk):
    zh = X @ Wh.T + bh
    a = sigmoid(zh)
    a = dropout(a, drop_ratio)
    zk = a @ Wk.T + bk
    b = np.maximum(zk, 0)
    b = dropout(b, drop_ratio)
    zo = b @ Wo.T + bo
    o = softmax(zo)
    return o


learningRate = 0.001
learningRate1 = 0.5
drop_ratio = 0.1
e = 1e-9
Epoch = 50
NumTrainSamples = 60000
NumTestSamples = 10000
NumInputs = 784
NumHiddenUnits = 512
NumClasses = 10
# inital weights
# hidden layer 1
Wh = np.matrix(np.random.uniform(-0.5, 0.5, (512, 784)))
bh = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh = np.zeros((NumHiddenUnits, NumInputs))
Wh_delta1 = np.zeros((NumHiddenUnits, NumInputs))
dbh = np.zeros((1, NumHiddenUnits))
# hidden layer 2
Wk = np.matrix(np.random.uniform(-0.5, 0.5, (512, 512)))
bk = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWk = np.zeros((NumHiddenUnits, NumHiddenUnits))
Wk_delta1 = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbk = np.zeros((1, NumHiddenUnits))
# Output layer
Wo = np.random.uniform(-0.5, 0.5, (10, 512))
bo = np.random.uniform(0, 0.5, (1, NumClasses))
dWo = np.zeros((NumClasses, NumHiddenUnits))
Wo_delta1 = np.zeros((NumClasses, NumHiddenUnits))
dbo = np.zeros((1, NumClasses))
# inital weights
# hidden layer 1
Wh1 = np.matrix(np.random.uniform(-0.5, 0.5, (512, 784)))
bh1 = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh1 = np.zeros((NumHiddenUnits, NumInputs))
dbh1 = np.zeros((1, NumHiddenUnits))
at1 = np.zeros((NumHiddenUnits, NumInputs))
# hidden layer 2
Wk1 = np.matrix(np.random.uniform(-0.5, 0.5, (512, 512)))
bk1 = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWk1 = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbk1 = np.zeros((1, NumHiddenUnits))
at2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
# Output layer
Wo1 = np.random.uniform(-0.5, 0.5, (10, 512))
bo1 = np.random.uniform(0, 0.5, (1, NumClasses))
dWo1 = np.zeros((NumClasses, NumHiddenUnits))
dbo1 = np.zeros((1, NumClasses))
at = np.zeros((NumClasses, NumHiddenUnits))
# inital weights
# hidden layer 1
Wh2 = np.matrix(np.random.uniform(-0.5, 0.5, (512, 784)))
bh2 = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh2 = np.zeros((NumHiddenUnits, NumInputs))
Wh_delta2 = np.zeros((NumHiddenUnits, NumInputs))
dbh2 = np.zeros((1, NumHiddenUnits))
# hidden layer 2
Wk2 = np.matrix(np.random.uniform(-0.5, 0.5, (512, 512)))
bk2 = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWk2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
Wk_delta2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbk2 = np.zeros((1, NumHiddenUnits))
# Output layer
Wo2 = np.random.uniform(-0.5, 0.5, (10, 512))
bo2 = np.random.uniform(0, 0.5, (1, NumClasses))
dWo2 = np.zeros((NumClasses, NumHiddenUnits))
Wo_delta2 = np.zeros((NumClasses, NumHiddenUnits))
dbo2 = np.zeros((1, NumClasses))
from IPython.display import clear_output

loss = []
Acc = []
loss1 = []
Acc1 = []
loss2 = []
Acc2 = []
Batch_size = 200
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range(0, NumTrainSamples, Batch_size):
        # feed fordware propagation
        Batch_samples = Stochastic_samples[ite:ite + Batch_size]
        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]
        zh = x @ Wh.T + bh
        a = sigmoid(zh)
        zk = a @ Wk.T + bk
        b = np.maximum(zk, 0)
        zo = b @ Wo.T + bo
        o = softmax(zo)
        # calculate cross entropy loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        # calculate back propagation error
        do = o - y
        dk = do @ Wo
        dh = dk @ Wk
        # update weight
        dWo = np.matmul(np.transpose(do), b)
        dks = np.multiply(dk, ReLU_derivative(zk))
        dWk = np.matmul(np.transpose(dks), a)
        dhs = np.multiply(np.multiply(dh, a), 1 - a)
        dWh = np.matmul(np.transpose(dhs), x)
        dbo = np.mean(do)
        dbk = np.mean(dks)
        dbh = np.mean(dhs)
        bo = bo - dbo
        bk = bk - dbk
        bh = bh - dbh
        Wo_delta1 = 0.95 * Wo_delta1 + learningRate * dWo / Batch_size
        Wo = Wo - Wo_delta1
        Wk_delta1 = 0.95 * Wk_delta1 + learningRate * dWk / Batch_size
        Wk = Wk - Wk_delta1
        Wh_delta1 = 0.95 * Wh_delta1 + learningRate * dWh / Batch_size
        Wh = Wh - Wh_delta1
        zh1 = x @ Wh1.T + bh1
        a1 = sigmoid(zh1)
        zk1 = a1 @ Wk1.T + bk1
        b1 = np.maximum(zk1, 0)
        zo1 = b1 @ Wo1.T + bo1
        o1 = softmax(zo1)
        # calculate cross entropy loss
        loss1.append(-np.sum(np.multiply(y, np.log10(o1))))
        # calculate back propagation error
        do1 = o1 - y
        dk1 = do1 @ Wo1
        dh1 = dk1 @ Wk1
        # update weight
        dWo1 = np.matmul(np.transpose(do1), b1)
        dks1 = np.multiply(dk1, ReLU_derivative(zk1))
        dWk1 = np.matmul(np.transpose(dks1), a1)
        dhs1 = np.multiply(np.multiply(dh1, a1), 1 - a1)
        dWh1 = np.matmul(np.transpose(dhs1), x)
        dbo1 = np.mean(do1)
        dbk1 = np.mean(dks1)
        dbh1 = np.mean(dhs1)
        bo1 = bo1 - dbo1
        bk1 = bk1 - dbk1
        bh1 = bh1 - dbh1
        at = at + np.power(dWo1, 2)
        n_delta = np.divide(np.multiply(learningRate1, dWo1 / Batch_size), np.sqrt(at + e))
        at2 = at2 + np.power(dWk1, 2)
        n_delta2 = np.divide(np.multiply(learningRate1, dWk1 / Batch_size), np.sqrt(at2 + e))
        at1 = at1 + np.power(dWh1, 2)
        n_delta1 = np.divide(np.multiply(learningRate1, dWh1 / Batch_size), np.sqrt(at1 + e))
        Wo1 = Wo1 - n_delta
        Wk1 = Wk1 - n_delta2
        Wh1 = Wh1 - n_delta1
        zh2 = x @ Wh2.T + bh2
        a2 = sigmoid(zh2)
        a2 = dropout(a2, drop_ratio)
        zk2 = a2 @ Wk2.T + bk2
        b2 = np.maximum(zk2, 0)
        b2 = dropout(b2, drop_ratio)
        zo2 = b2 @ Wo2.T + bo2
        o2 = softmax(zo2)
        # calculate cross entropy loss
        loss2.append(-np.sum(np.multiply(y, np.log10(o2))))
        # calculate back propagation error
        do2 = o2 - y
        dk2 = do2 @ Wo2
        dh2 = dk2 @ Wk2
        # update weight
        dWo2 = np.matmul(np.transpose(do2), b2)
        dks2 = np.multiply(dk2, ReLU_derivative(zk2))
        dWk2 = np.matmul(np.transpose(dks2), a2)
        dhs2 = np.multiply(np.multiply(dh2, a2), 1 - a2)
        dWh2 = np.matmul(np.transpose(dhs2), x)
        dbo2 = np.mean(do2)
        dbk2 = np.mean(dks2)
        dbh2 = np.mean(dhs2)
        bo2 = bo2 - dbo2
        bk2 = bk2 - dbk2
        bh2 = bh2 - dbh2
        Wo_delta2 = 0.95 * Wo_delta2 + learningRate * dWo2 / Batch_size
        Wo2 = Wo2 - Wo_delta2
        Wk_delta2 = 0.95 * Wk_delta2 + learningRate * dWk2 / Batch_size
        Wk2 = Wk2 - Wk_delta2
        Wh_delta2 = 0.95 * Wh_delta2 + learningRate * dWh2 / Batch_size
        Wh2 = Wh2 - Wh_delta2
    # Test accuracy with random innitial weights
    prediction = Forwardpass(x_test, Wh, bh, Wo, bo, Wk, bk)
    Acc.append(AccTest(y_test, prediction))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)], Acc, "go", label='Momentum')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy test")
    plt.title("Compare accuracy three kind of optimization")
    # Test accuracy with random innitial weights
    prediction1 = Forwardpass(x_test, Wh1, bh1, Wo1, bo1, Wk1, bk1)
    Acc1.append(AccTest1(y_test, prediction1))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc1)], Acc1, "o", label='ADAGRAD')
    # Test accuracy with random innitial weights
    prediction2 = Forwardpass1(x_test, Wh2, bh2, Wo2, bo2, Wk2, bk2)
    Acc2.append(AccTest2(y_test, prediction2))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc2)], Acc2, "ro", label='Dropout + Momentum')
    plt.legend(loc='best')
    print('Epoch:', ep)
    print('Accuracy1:', AccTest(y_test, prediction))
    print('Accuracy2:', AccTest1(y_test, prediction1))
    print('Accuracy3:', AccTest2(y_test, prediction2))
    plt.show()
