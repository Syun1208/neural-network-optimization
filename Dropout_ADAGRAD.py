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
