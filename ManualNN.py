import pandas
from ucimlrepo import fetch_ucirepo
import numpy as np
import math

powerplant = fetch_ucirepo(id=294)
dataset = powerplant.data

class activationFunction:
    def __init__(self, act, deriv) -> None:
        self.act = act
        self.der = deriv
        pass

class Neuron:
    #Previous layer is a list of inputNeurons
    def __init__(self, actFunction):
        #Initialize all n (+1 for the bias) weights in this neuron upon creation
        self.weights = [0]
        self.actFun = actFunction

    def initWeights(self,prevLayerSize):
        self.weights = np.random.rand(prevLayerSize + 1)

    def calcOutput(self, input):
        a = 0
        for i,inputFrom in enumerate(input):
            # print(self.weights[i+1], "*" , inputFrom," + ", a ,"=")
            a += inputFrom * self.weights[i+1]
            # print(a)

        return self.actFun.act(a + self.weights[0])
    
    def __str__(self):
        return "(weights: " +  str(self.weights) + ")"

def makeNNforDataset(dataset,activationFunction):
    name = dataset.metadata["name"]
    features = dataset.metadata["num_features"]
    target = dataset.metadata["target_col"]

    hiddenlayerSizes = [3,2] # output layer is implied

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
    
    #init output
    outputNeuron = Neuron(activationFunction)
    outputNeuron.initWeights(Nn[-1].__len__())
    Nn.append([outputNeuron])
    
    return Nn

def getDataRow(data,row):
    return data.features.iloc[row].to_numpy()

def forwardPass(Nn,data,row):
    dataRow = getDataRow(data,row)

    NnOut = []

    for l,layer in enumerate(Nn):
        neuronInput = []
        if l == 0:
            neuronInput = dataRow
        else:
            neuronInput = NnOut[l-1]
        NnOut.append([])
        for n in layer:
            NnOut[l].append(n.calcOutput(neuronInput))
            pass

    return NnOut

def getTargetData(data,row):
    return data.targets.iloc[row].to_numpy()
    
def backwardsPass(Nn,data,outputs,targetRow,learningRate):
    delta = []
    target = getTargetData(data,targetRow)
    for i,outputNeuon in enumerate(Nn[-1]):
        out = outputs[-1][i]
        deltaValue = outputNeuon.actFun.der(out) * (target - out)
        delta.append(deltaValue)
    pass

    for l, hiddenLayer in reversed(list(enumerate(Nn[:-1]))):
        delta.insert(0,[])
        for i, neuron in enumerate(hiddenLayer):
            weightDeltaSum = 0
            for k,forwardNeuron in enumerate(Nn[l+1]):
                weightDeltaSum += forwardNeuron.weights[i+1] * delta[1][k]
            deltaValue = neuron.actFun.der(outputs[l][i]) * weightDeltaSum
            delta[0].append(deltaValue)
        pass
    pass

    print(Nn[0][0].weights)
    Nn[0][0].weights[1] += learningRate * delta[1][0] * outputs[1][0]
    print(Nn[0][0].weights)

    # for l, layer in enumerate(Nn):
    #     print("**********",l)
    #     for j,neuron in enumerate(layer):
    #         print("----",j)

    #         neuron.weights[0] += learningRate * delta[l][j]
    #         for i in range(1,neuron.weights.__len__()):
    #             # print("b4 ",w , learningRate,delta[l][j],outputs[l][j])
    #             print(i)
    #             neuron.weights[i] += learningRate * delta[l][j] * outputs[l-1][j]
    #             # print("aft",neuron.weights[i])
    
    # print(delta,outputs,target)
    pass



def sigmoid_act(net_sum):
    sigmoid_act1 = 1 / (1 + math.exp(-net_sum)) 
    return sigmoid_act1

def sigmoid_der(sigmoid_act1):
    # sigmoid_act1 = sigmoid_act(net_sum)
    sigmoid_der1 = sigmoid_act1 * (1 - sigmoid_act1)
    return sigmoid_der1

def tanh_act(net_sum):
    tanh_act1 = (math.exp(net_sum) - math.exp(-net_sum))/(math.exp(net_sum) + math.exp(-net_sum))
    return tanh_act1

def tanh_der(tanh_act1):
    # tanh_act1 = tanh_act(net_sum)
    tanh_der1 = 1 - tanh_act1 ** 2
    return tanh_der1

def relu_act(net_sum):
    if net_sum > 0:
        relu_act1 = net_sum
    else:
        relu_act1 = 0
    return relu_act1

def relu_der(neuronOutput):
    if neuronOutput > 0:
        relu_der1 = 1
    else:
        relu_der1 = 0
    return relu_der1

sigmoid = activationFunction(sigmoid_act,sigmoid_der)
tanh = activationFunction(tanh_act,tanh_der)
relu = activationFunction(relu_act,relu_der)

net = makeNNforDataset(powerplant,relu)
nnOutputs = forwardPass(net,dataset,0)
backwardsPass(net,dataset,nnOutputs,0,0.1)