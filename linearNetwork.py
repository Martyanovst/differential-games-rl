import numpy as np
import torch
from torch import nn
import random
import gym
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.autograd import Variable

class Identical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class LinearNetwork(nn.Module):
    def __init__(self, layers, hidden_activation, output_activation):
        super().__init__()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layers_count = len(layers)
        self.output_layer = nn.Linear(
            layers[self.layers_count - 2], layers[self.layers_count - 1])
        self.init_hidden_layers_(layers)
        self.apply(self._init_weights_)

    def init_hidden_layers_(self, layers):
        self.hidden_layers = []
        for i in range(1, len(layers) - 1):
            previous_layer = layers[i - 1]
            current_layear = layers[i]
            linear = nn.Linear(previous_layer, current_layear)
            self.hidden_layers.append(linear)

    def forward(self, tensor):
        hidden = tensor
        for layer in self.hidden_layers:
            hidden = self.hidden_activation(layer(hidden))
        output = self.output_activation(self.output_layer(hidden))
        return output

    def _init_weights_(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)