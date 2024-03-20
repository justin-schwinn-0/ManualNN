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

class Neuron:
    #Previous layer is a list of inputNeurons
    def __init__(self, actFunction):
        #Initialize all n (+1 for the bias) weights in this neuron upon creation
        self.bias = 0
        self.weights = [0]
        self.actFun = actFunction

    def initWeights(self,prevLayerSize):
        self.weights = np.random.rand(prevLayerSize)

    def calcOutput(self, input):
        a = 0
        for i,inputFrom in enumerate(input):
            a += inputFrom * self.weights[i]

        return self.actFun.act(a) #TODO add bias
    
    def str(self):
        return "(weights: " +  str(self.weights) + ")"

def makeNNforDataset(dataset,activationFunction):
    name = dataset.metadata["name"]
    features = dataset.metadata["num_features"]
    target = dataset.metadata["target_col"]

    print("Making NN for Dataset: ", name)
    print("using", features ,"input nodes")

    hiddenlayerSizes = [3] # 

    Nn = []

    #init hidden layer(s)
    for l,size in enumerate(hiddenlayerSizes):
        Nn.append([])
        for i in range(size): # hardcode input for dataset
            Nn[l].append(Neuron(activationFunction))
            if l == 0:
                Nn[l][i].initWeights(features)
            else:
                Nn[l][i].initWeights(hiddenlayerSizes[l-1])
            print("layer",l,Nn[l][i].str())
    
    #init output
    outputNeuron = Neuron(activationFunction)
    outputNeuron.initWeights(Nn[-1].__len__())
    print("output:", outputNeuron.str()) 
    Nn.append([outputNeuron])

    for layer in Nn:
        print("layer with", layer.__len__(), "Neurons")
    
    return Nn

def forwardPass(Nn,data):

    pass


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

net = makeNNforDataset(dataset,relu)