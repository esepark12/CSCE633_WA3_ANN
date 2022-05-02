import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

## dimensions [size][x/y][784][1]
#n = ### YOUR CODE HERE ###
from network import Network
n = (825006152 % 24) + 4
#net = net.Network([784, n, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=validation_data)


#########################Q1: tuning hyperparameters
import math
import gc
#Tune mini batch size
'''
mini_batches = [10, 20, 30] #list of mini-batch sizes
best_acc = -math.inf
for mini_batch in mini_batches:
    print(f"Mini batch size: {mini_batch}")
    net = Network([784, n, 10])
    accuracies = net.SGD(training_data, 30, mini_batch, 3.0, test_data=validation_data)
    if accuracies['test_acc'][29] > best_acc:
        best_acc = accuracies['test_acc'][29]
    del net
    del accuracies
    #gc.collect()
print(f"Best accuracy: {best_acc}")
'''
#Tune learning rate
'''
lrs = [1.0, 3.0, 5.0, 10.0] #list of mini-batch sizes
best_lr = -math.inf
for lr in lrs:
    print(f"Learning rate: {lr}")
    net = Network([784, n, 10])
    accuracies = net.SGD(training_data, 30, 10, lr, test_data=validation_data)
    if accuracies['test_acc'][29] > best_lr:
        best_lr = accuracies['test_acc'][29]
    del net
    del accuracies
    #gc.collect()
print(f"Best accuracy: {best_acc}")
'''

