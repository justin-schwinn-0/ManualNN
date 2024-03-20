import pandas
from ucimlrepo import fetch_ucirepo
import numpy as np

dataset = fetch_ucirepo(id=294)

class activationFunction:
    def __init__(self, act, deriv) -> None:
        self.activation = act
        self.derivitive = deriv
        pass

class perceptron:
    def __init__(self) -> None:
        
        pass

def test1Act():
    print("test1 act")

def test1der():
    print("test1 der")

def makeNNforDataset(dataset):
    print(dataset.metadata)
    name = dataset.metadata["name"]
    features = dataset.metadata["num_features"]
    target = dataset.metadata["target_col"]
    print("Making NN for Dataset: ", name)
    print("using", features ,"input nodes")

test1 = activationFunction(test1Act,test1der)

test1.activation()
test1.derivitive()

print

makeNNforDataset(dataset)