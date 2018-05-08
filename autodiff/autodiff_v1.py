#!/usr/bin/env python

import sys
import math
from collections import namedtuple


Child = namedtuple("Child", "node grad")


class Variable(object):
    """
    Simple version.
    Forward pass is done once at the initialization.
    """
    def __init__(self, value):
        self.children = []
        self.parents = []
        self.value = value
        self._grad = None

    def eval(self):
        return self.value

    def __add__(self, other):
        z = Variable(self.value + other.value)
        self.children.append(Child(z, 1))
        other.children.append(Child(z, 1))
        z.parents.extend([self, other])
        return z

    def __mul__(self, other):
        z = Variable(self.value * other.value)
        self.children.append(Child(z, other.value))
        other.children.append(Child(z, self.value))
        z.parents.extend([self, other])
        return z

    def zero_grad(self):
        self._grad = 0

    @property
    def grad(self):
        if self._grad is not None:
            return self._grad
        if not self.children:
            return 1.0
        else:
            self._grad = sum([child.node.grad * child.grad for child in self.children])
            return self._grad

def zero_grad(root):
    root.zero_grad()
    for p in root.parents:
        p.zero_grad()


def exp(x):
    z = Variable(math.exp(x.value))
    x.children.append(Child(z, math.exp(x.value)))
    z.parents.append(x)
    return z


def sin(x):
    z = Variable(math.sin(x.value))
    x.children.append(Child(z, math.cos(x.value)))
    z.parents.append(x)
    return z


def cos(x):
    z = Variable(math.cos(x.value))
    x.children.append(Child(z, -math.cos(x.value))) 
    z.parents.append(x)
    return z


def log(x):
    z = Variable(math.log(x.value))
    x.children.append(Child(z, 1 / x.value))
    z.parents.append(x)
    return z


if __name__ == "__main__":

    # f = exp(x) + x * y = a + b
    # fx = y + exp(x)
    # fy = x
    # fa = 1
    # fb = 1
    x = Variable(1.0)
    y = Variable(2.0)
    a = exp(x)
    b = x * y
    z = a + b

    print a.eval(), b.eval()
    print "dz/da, dz/db", a.grad, b.grad
    print "dz/dx, dz/dy", x.grad, y.grad
    print z.eval()

