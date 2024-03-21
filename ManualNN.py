import pandas
from ucimlrepo import fetch_ucirepo
import numpy as np
import math

powerplant = fetch_ucirepo(id=294)

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

def makeNNforDataset(dataset,activationFunction,LayersArray):
    name = dataset.metadata["name"]
    features = dataset.metadata["num_features"]
    target = dataset.metadata["target_col"]

    hiddenlayerSizes = LayersArray # output layer is implied

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
    
def backwardsPass(Nn,data,targetRow,learningRate):
    inputs= getDataRow(data,targetRow)
    outputs = forwardPass(net,dataset,targetRow)
    target = getTargetData(data,targetRow)
    
    delta = []
    for i,outputNeuron in enumerate(Nn[-1]):
        out = outputs[-1][i]
        deltaValue = outputNeuron.actFun.der(out) * (target - out)
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

    for layer in range(len(Nn)):
        for neuron in range(len(Nn[layer])):
            Nn[layer][neuron].weights[0] += learningRate * delta[layer][neuron]
            for weightIndex in range(1,len(Nn[layer][neuron].weights)):
                if layer == 0:
                    Nn[layer][neuron].weights[weightIndex] += learningRate * delta[layer][neuron] * inputs[weightIndex-1]
                else:
                    Nn[layer][neuron].weights[weightIndex] += learningRate * delta[layer][neuron] * outputs[layer-1][weightIndex-1]
                pass
            pass
        pass
    pass

    return outputs[-1][-1],target

def epoch(Nn,data,begin,end,learningRate):
    for i in range(begin,end):
        out,target = backwardsPass(Nn,data,i,learningRate)
        #print("Prediction:",out,"actual:",target, "delta:",(target-out))
    pass

def squarederror(Nn,dataset, numtraining, numtest):
    a = 0
    for elementIndex in range(numtraining, len(dataset.features)):
        outputs = forwardPass(Nn,dataset,elementIndex)
        #print(getTargetData(dataset, elementIndex) - outputs[-1][0])
        a += (getTargetData(dataset, elementIndex) - outputs[-1][0])**2

    return (a / numtest)

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

dataset = powerplant.data


numTraining = math.ceil(len(dataset.features) * .7)
numTest = len(dataset.features) - numTraining


scaledData = dataset.copy()
print(dataset.features.iloc[0])
print(dataset.targets.iloc[0])
columns = ['AT','V','AP','RH']
for col in columns:
    scaledData["features"][col] = scaledData["features"][col] / scaledData["features"][col].abs().max()
scaledData["targets"]["PE"] = scaledData["targets"]["PE"] / scaledData["targets"]["PE"].abs().max()

net = makeNNforDataset(powerplant,relu,[5,5])
epoch(net,dataset,0,numTraining,0.001)
print("sqr Erorr:",squarederror(net,dataset, numTraining, numTest))