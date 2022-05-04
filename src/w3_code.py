import mnist_loader
import network

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

## dimensions [size][x/y][784][1]
#n = ### YOUR CODE HERE ###
from network import Network
n = (825006152 % 24) + 4
#net = net.Network([784, n, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=validation_data)


#########################Q1: tuning hyperparameters
import math
import numpy as np
import gc

#Tune mini batch size
best_batch = 9
'''
mini_batches = [8, 9, 10] #list of mini-batch sizes
best_acc = -math.inf
best_batch = mini_batches[0]
for mini_batch in mini_batches:
    print(f"Mini batch size: {mini_batch}")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, n, 10])
    accuracies = net.SGD(training_data, 30, mini_batch, 3.0, test_data=validation_data)
    if np.mean(accuracies['test_acc']) > best_acc:
        best_acc = np.mean(accuracies['test_acc'])
        best_batch = mini_batch
    del net
    del accuracies
    #gc.collect()
print(f"Best accuracy: {best_acc}")
print(f"Best batch: {best_batch}")
'''
#Tune learning rate
best_lr = 1.0
'''
lrs = [1.0, 2.0, 3.0, 4.0] #list of mini-batch sizes
best_acc2 = -math.inf
best_lr = lrs[0]
for lr in lrs:
    print(f"Learning rate: {lr}")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, n, 10])
    accuracies = net.SGD(training_data, 30, best_batch, lr, test_data=validation_data)
    if np.mean(accuracies['test_acc']) > best_acc2:
        best_acc2 = np.mean(accuracies['test_acc'])
    del net
    del accuracies
    #gc.collect()
print(f"Best accuracy: {best_acc2}")
print(f"Best learning rate: {best_lr}")
'''
#test with best hyperparameters
'''
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, n, 10])
accuracies = net.SGD(training_data, 30, best_batch, best_lr, test_data=validation_data)
best_acc = np.mean(accuracies['test_acc'])
print(f"Best accuracy: {best_acc}")
'''
###################################Q2: Bagging
import sklearn
from sklearn.utils import resample
from matplotlib import pyplot
test_accuracies = []
train_accuracies = []
m_list = [1,2,5,10,20]
for m in m_list:

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    ensemble = 3 #ensemble size
    testscores = [] #average test accuracies of models
    trainscores = []
    models = []
    train = list(training_data)
    test = list(test_data)
    for e in range(ensemble):
        #get split index
        ix = [i for i in range(len(train))]
        train_ix = resample(ix, replace=True, n_samples=40000)
        test_ix = [x for x in ix if x not in train_ix]
        #split data
        newtrain = [train[index] for index in train_ix]
        newtest = [train[index] for index in test_ix]
        #newtest = train
        #train model&get test accuracy
        net = Network([784, n, 10])
        accuracies = net.SGD(newtrain, 5, best_batch, best_lr, test_data=newtest)
        models.append(net)
        testscores.append(np.mean(accuracies['test_acc'])/len(newtest))
        trainscores.append(np.mean(accuracies['train_acc'])/len(newtrain))
    print(f"Best test accuracy: {np.mean(testscores)}")
    print(f"Best train accuracy: {np.mean(trainscores)}")
    test_accuracies.append(np.mean(testscores))
    train_accuracies.append(np.mean(trainscores))
pyplot.plot(m_list, test_accuracies, marker='o', markerfacecolor='blue')
pyplot.plot(m_list, train_accuracies, marker='o', markerfacecolor='red')
pyplot.show()