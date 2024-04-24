import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import dreamerv2.models.vdp as vdp

class DenseModel(nn.Module):
    def __init__(
            self, 
            output_shape,
            input_size, 
            info,
            input_flag=False,
            output_flag=False
        ):
        """
        :param output_shape: tuple containing shape of expected output
        :param input_size: size of input features
        :param info: dict containing num of hidden layers, size of hidden layers, activation function, output distribution etc.
        """
        super().__init__()
        self._output_shape = output_shape
        self._input_size = input_size
        self._layers = info['layers']
        self._node_size = info['node_size']
        self.activation = info['activation']
        self.dist = info['dist']
        self.model = self.build_model(input_flag=input_flag, output_flag=output_flag)

    def build_model(self, input_flag = False, output_flag=False):
        model = [vdp.Linear(self._input_size, self._node_size, input_flag=input_flag, tuple_input_flag=True)]
        model += [self.activation(tuple_input_flag=True)]
        for i in range(self._layers-1):
            model += [vdp.Linear(self._node_size, self._node_size, tuple_input_flag=True)]
            model += [self.activation(tuple_input_flag=True)]
        model += [vdp.Linear(self._node_size, int(np.prod(self._output_shape)), output_flag=output_flag, tuple_input_flag=True)]
        return nn.Sequential(*model)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        dist_mu, dist_sigma = self.model((mu_x, sigma_x))
        if self.dist == 'normal':
            return td.independent.Independent(td.Normal(dist_mu, dist_sigma), len(self._output_shape))
        if self.dist == 'binary':
            logits = dist_mu + torch.sqrt(dist_sigma)*torch.rand_like(dist_sigma)
            return td.independent.Independent(td.Bernoulli(logits=logits), len(self._output_shape))
        if self.dist == None:
            return dist_mu, dist_sigma

        raise NotImplementedError(self._dist)