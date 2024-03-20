import pandas
from ucimlrepo import fetch_ucirepo
import numpy as np

iris = fetch_ucirepo(id=53)


class activationFunction:
    def __init__(self, act, deriv) -> None:
        self.activation = act
        self.derivitive = deriv
        pass

def test1Act():
    print("test1 act")

def test1der():
    print("test1 der")


test1 = activationFunction(test1Act,test1der)

test1.activation()
test1.derivitive()