import sympy as sym
from typing import List
import torch
import torch.nn as nn


def compute_layers(params_number: int, varying_layer: int,  input_size: int, fixed_layer_size: int, output_size: int) ->List[int]:
        '''
        Function to compute varying layer values. Works with only two hidden layers at the moment.
        :param params_number: total number of parameters
        :parma varying_layer: position of varying layer,  with 1 corresponding to
        the first layer, and 2, to the second.
        :param input_size: input size
        :param fixed_layer_size: fixed layer size/width
        :param output_size: output layer size
        :return:
        '''

        p, u0, u1, u2, u3 = sym.symbols('p u0 u1 u2 u3')
        layers = [u0, u1, u2, u3]

        solve_for_layer = sym.solveset((u0*u1 + u1)+ (u1*u2 + u2) + ( u2*u3+ u3) - p,
                                       layers[varying_layer])
        variables = [p]+ layers[:varying_layer]+layers[varying_layer+1:]
        expression = sym.lambdify(variables, solve_for_layer, modules=['math'])
        numerical_solution = expression(params_number, input_size, fixed_layer_size, output_size)
        layer_size =  int(numerical_solution)
        assert layer_size >= 1, 'must be >= 1, reduce fixed_layer_size or increase params_number'

        layers_expression = sym.lambdify(layers, layers, modules=['math'])

        if varying_layer == 1:
            return layers_expression(input_size, layer_size, fixed_layer_size, output_size)
        else:
            return layers_expression(input_size, fixed_layer_size, layer_size, output_size)


class Net(nn.Module):
    def __init__(self, layers : List[int], initialization : str):
        '''

        :param layers: list with input, first, second, and output layer values
        :param initialization: 'xavier' or 'kaiming'
        '''

        [u0, u1, u2, u3] = layers

        super().__init__()
        self.fc1 = nn.Linear(u0, u1)
        if initialization == 'kaiming':
            nn.init.kaiming_uniform_(self.fc1.weight, a= np.sqrt(2), mode='fan_in', nonlinearity='leaky_relu')
        else:
            nn.init.xavier_uniform_(self.fc1.weight, gain= 1/np.sqrt(3))
        self.bn1 = nn.BatchNorm1d(num_features=u1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc2 = nn.Linear(u1, u2)
        self.bn2 = nn.BatchNorm1d(num_features=u2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if initialization == 'kaiming':
           nn.init.kaiming_uniform_(self.fc2.weight, a= np.sqrt(2), mode='fan_in', nonlinearity='leaky_relu')
        else:
            nn.init.xavier_uniform_(self.fc2.weight, gain= 1/np.sqrt(3))

        self.fc3 = nn.Linear(u2, u3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x







