import pandas
from ucimlrepo import fetch_ucirepo
import numpy as np
import math

dataset = fetch_ucirepo(id=294)

class activationFunction:
    def __init__(self, act, deriv) -> None:
        self.activation = act
        self.derivative = deriv
        pass

def test1Act():
    print("test1 act")

def test1der():
    print("test1 der")

class Neuron:
    #Previous layer is a list of inputNeurons
    def init(self, actFunction):
        #Initialize all n (+1 for the bias) weights in this neuron upon creation
        self.bias = 0
        self.weights = {0}
        self.actFun = actFunction

    def calcOutput(self, input):
        a = 0
        for i,inputFrom in enumerate(input):
            a += inputFrom * self.weights[i] + self.bias

        return self.actFun.act(a)

def makeNNforDataset(dataset,activationFunction):
    name = dataset.metadata["name"]
    features = dataset.metadata["num_features"]
    target = dataset.metadata["target_col"]

    print("Making NN for Dataset: ", name)
    print("using", features ,"input nodes")

    layers = 1 # 1 hidden. input and output is implied

    layerSizes = {4}

    Nn = [[]]
    #init input
    Nn[0].append([]) 
    for i in range(features): # hardcode input for dataset
        print("make neuron",i)
        Nn[0][i] = Neuron(activationFunction)


    #init hidden layer(s)

    #init output




    # NN  should be an array of n=features+1 elements for the first layer
    # NN [layer][nueron] jagged array

def sigmoid_act(net_sum):
    sigmoid_act1 = 1 / (1 + math.exp(-net_sum)) 
    return sigmoid_act1

def sigmoid_der(net_sum):
    sigmoid_act1 = sigmoid_act(net_sum)
    sigmoid_der1 = sigmoid_act1 * (1 - sigmoid_act1)
    return sigmoid_der1

def tanh_act(net_sum):
    tanh_act1 = (math.exp(net_sum) - math.exp(-net_sum))/(math.exp(net_sum) + math.exp(-net_sum))
    return tanh_act1

def tanh_der(net_sum):
    tanh_act1 = tanh_act(net_sum)
    tanh_der1 = 1 - tanh_act1 ** 2
    return tanh_der1

def relu_act(net_sum):
    if net_sum > 0:
        relu_act1 = net_sum
    else:
        relu_act1 = 0
    return relu_act1

def relu_der(net_sum):
    if net_sum > 0:
        relu_der1 = 1
    else:
        relu_der1 = 0
    return relu_der1

sigmoid = activationFunction(sigmoid_act,sigmoid_der)
tanh = activationFunction(tanh_act,tanh_der)
relu = activationFunction(relu_act,relu_der)


test1 = activationFunction(test1Act,test1der)

test1.activation()
test1.derivative()

print

makeNNforDataset(dataset,relu)