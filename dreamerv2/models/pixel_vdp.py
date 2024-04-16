import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import vdp


class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size, info):
        """
        :param input_shape: tuple containing shape of input
        :param embedding_size: Supposed length of encoded vector
        """
        super(ObsEncoder, self).__init__()
        self.shape = input_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        #print(k)
        self.k = k
        self.d = d
        self.convolutions = nn.Sequential(
            vdp.Conv2d(input_shape[0], d, k, padding=1, input_flag=True),
            activation(tuple_input_flag=True),
            vdp.Conv2d(d, 2*d, k, padding=1, tuple_input_flag=True),
            activation(tuple_input_flag=True),
            vdp.Conv2d(2*d, 4*d, k, padding=1, tuple_input_flag=True),
            activation(tuple_input_flag=True),
        )
        #print(self.embed_size)
        if embedding_size == self.embed_size:
            self.fc_1 = vdp.Identity()
        else:
            self.fc_1 = vdp.Linear(self.embed_size, embedding_size)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed_mu, embed_sigma = self.convolutions(obs.reshape(-1, *img_shape))
        embed_mu = torch.reshape(embed_mu, (*batch_shape, -1))
        embed_sigma = torch.reshape(embed_sigma, (*batch_shape, -1))
        embed_mu, embed_sigma = self.fc_1(embed_mu, embed_sigma)
        return embed_mu, embed_sigma

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 1, self.k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 1, self.k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 1, self.k, 1)
        #print(conv1_shape, conv2_shape, conv3_shape)
        embed_size = int(4*self.d*np.prod(conv3_shape).item())
        return embed_size

class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, info):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        conv1_shape = conv_out_shape(output_shape[1:], 1, k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 1, k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 1, k, 1)
        self.conv_shape = (4*d, *conv3_shape)
        self.output_shape = output_shape
        if embed_size == np.prod(self.conv_shape).item():
            self.linear = vdp.Identity()
        else:
            self.linear = vdp.Linear(embed_size, np.prod(self.conv_shape).item())
        self.decoder = nn.Sequential(
            vdp.ConvTranspose2d(4*d, 2*d, k, 1, padding=1),
            activation(tuple_input_flag=True),
            vdp.ConvTranspose2d(2*d, d, k, 1, padding=1, tuple_input_flag=True),
            activation(tuple_input_flag=True),
            vdp.ConvTranspose2d(d, c, k, 1, padding=1, tuple_input_flag=True),
        )

    def forward(self, mu_x, sigma_x):
        batch_shape = mu_x.shape[:-1]
        embed_size = mu_x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        mu_x = mu_x.reshape(squeezed_size, embed_size)
        sigma_x = sigma_x.reshape(squeezed_size, embed_size)
        mu_x, sigma_x = self.linear(mu_x, sigma_x)
        mu_x = torch.reshape(mu_x, (squeezed_size, *self.conv_shape))
        sigma_x = torch.reshape(sigma_x, (squeezed_size, *self.conv_shape))
        mu_x, sigma_x = self.decoder(mu_x, sigma_x)
        mean = torch.reshape(mu_x, (*batch_shape, *self.output_shape))
        var = torch.reshape(sigma_x, (*batch_shape, *self.output_shape))
        obs_dist = td.Independent(td.Normal(mean, var), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
