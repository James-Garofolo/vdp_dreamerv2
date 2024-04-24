import math
import torch
import torch.nn.functional as F
import numpy as np

def softplus(a):
    return torch.log(1.+torch.exp(torch.clamp(a, min=-88, max=88)))

def i_softplus(a):
    return torch.clamp(torch.log(torch.exp(a-1)), min=-88, max=88)


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Identity, self).__init__(*args, **kwargs)

    def forward(self, mu_x, sigma_x):
        return mu_x, sigma_x

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, input_flag=False, output_flag=False, tuple_input_flag=False):
        super(Linear, self).__init__()
        self.input_flag = input_flag
        self.tuple_input_flag = tuple_input_flag
        self.bias = bias
        self.mu = torch.nn.Linear(in_features, out_features, bias)
        self.sigma = torch.nn.Linear(in_features, out_features, bias)
        if output_flag:
            torch.nn.init.zeros_(self.mu.weight)
            torch.nn.init.ones_(self.sigma.weight)
        else:
            torch.nn.init.xavier_normal_(self.mu.weight)
            torch.nn.init.uniform_(self.sigma.weight, a=0, b=5)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        if torch.any(torch.isnan(self.mu.weight)) or torch.any(torch.isnan(self.sigma.weight)):
            raise ValueError("nans from within")
        if torch.any(torch.isinf(self.mu.weight)) or torch.any(torch.isinf(self.sigma.weight)):
            raise ValueError("infs from within")
        if torch.any(self.sigma.weight == 0):
            raise ValueError("0s from within")
        if self.input_flag:
            mu_y = self.mu(mu_x)
            sigma_y = mu_x ** 2 @ softplus(self.sigma.weight).T
            pass
        else:
            if self.tuple_input_flag:
                mu_x, sigma_x = mu_x
            mu_y = self.mu(mu_x)
            if len(sigma_x.shape) > 2:
                sigma_y = (softplus(self.sigma.weight) @ sigma_x.mT).mT + \
                          (self.mu.weight**2 @ sigma_x.mT).mT + \
                          (mu_x ** 2 @ softplus(self.sigma.weight).mT)
            else:
                sigma_y = (softplus(self.sigma.weight) @ sigma_x.T).T + \
                        (self.mu.weight**2 @ sigma_x.T).T + \
                        (mu_x ** 2 @ softplus(self.sigma.weight).T)
            pass
        if self.bias:
            sigma_y += softplus(self.sigma.bias)
        sigma_y *= 1e-3
        return mu_y, sigma_y

    def kl_term(self):
        kl = 0.5*torch.mean(self.mu.weight.shape[1] * softplus(self.sigma.weight) + torch.norm(self.mu.weight)**2
                            - self.mu.weight.shape[1] - self.mu.weight.shape[1] * torch.log(softplus(self.sigma.weight)))
        return kl

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(5, 5), stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros',
                 input_flag=False, tuple_input_flag=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.input_flag = input_flag
        self.tuple_input_flag = tuple_input_flag
        self.mu = torch.nn.Conv2d(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation,
                                  groups, bias, padding_mode)
        torch.nn.init.xavier_normal_(self.mu.weight)
        self.sigma = torch.nn.Linear(1, out_channels, bias=bias)
        torch.nn.init.uniform_(self.sigma.weight, a=0, b=5)

        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        if self.input_flag:
            mu_y = self.mu(mu_x)
            vec_x = self.unfold(mu_x)
            sigma_y = softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ vec_x ** 2
            sigma_y = sigma_y.view(mu_y.shape[0], mu_y.shape[1], mu_y.shape[2], mu_y.shape[3])
            pass
        else:
            if self.tuple_input_flag:
                mu_x, sigma_x = mu_x
            mu_y = self.mu(mu_x)
            vec_x = self.unfold(mu_x)
            sigma_y = (softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ self.unfold(sigma_x)) + \
                      (self.mu.weight.view(self.out_channels, -1)**2 @ self.unfold(sigma_x).permute(2, 1, 0)).permute(2, 1, 0) + \
                      (softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ (vec_x ** 2).permute(2, 1, 0)).permute(2, 1, 0)
            sigma_y = sigma_y.view(mu_y.shape[0], mu_y.shape[1], mu_y.shape[2], mu_y.shape[3])
        sigma_y *= 1e-3
        return mu_y, sigma_y

    def kl_term(self):
        kl = 0.5*torch.mean(self.kernel_size * softplus(self.sigma.weight) + torch.norm(self.mu.weight)**2
                            - self.kernel_size - self.kernel_size * torch.log(softplus(self.sigma.weight)))
        return kl
    
class ConvTranspose2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=1, 
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, 
                 padding_mode='zeros', input_flag=False, tuple_input_flag=False):
        
        super(ConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.input_flag = input_flag
        self.tuple_input_flag = tuple_input_flag
        self.mu = torch.nn.ConvTranspose2d(in_channels, out_channels,
                                  kernel_size, stride, padding, output_padding,
                                  groups, bias, dilation, padding_mode)
        torch.nn.init.xavier_normal_(self.mu.weight)
        self.sigma = torch.nn.Linear(1, out_channels, bias=bias)
        torch.nn.init.uniform_(self.sigma.weight, a=0, b=5)

        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

    def vk(self):
        """var kernel"""
        if (type(self.kernel_size) == tuple) or (type(self.kernel_size) == list):
            return self.sigma.weight.view(1,-1,1,1).repeat(self.in_channels, 1, 
                                    self.kernel_size[0], self.kernel_size[1])
        else:
            return self.sigma.weight.view(1,-1,1,1).repeat(self.in_channels, 1, 
                                    self.kernel_size, self.kernel_size)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        if self.input_flag:
            mu_y = self.mu(mu_x)
            #vec_x = self.unfold(mu_x)
            #sigma_y = softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ vec_x ** 2
            #sigma_y = sigma_y.view(mu_y.shape[0], mu_y.shape[1], mu_y.shape[2], mu_y.shape[3])
            sigma_y = F.conv_transpose2d(mu_x**2, softplus(self.vk()), None, self.stride,
                                         self.padding, self.output_padding, 1, self.dilation)
            pass
        else:
            if self.tuple_input_flag:
                mu_x, sigma_x = mu_x
            mu_y = self.mu(mu_x)

            sigma_y = F.conv_transpose2d(mu_x**2, softplus(self.vk()), None, self.stride,
                                         self.padding, self.output_padding, 1, self.dilation) +\
                      F.conv_transpose2d(sigma_x, self.mu.weight**2, None, self.stride,
                                         self.padding, self.output_padding, 1, self.dilation) +\
                      F.conv_transpose2d(sigma_x, softplus(self.vk()), None, self.stride,
                                         self.padding, self.output_padding, 1, self.dilation)
            #sigma_y = (softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ self.unfold(sigma_x)) + \
            #          (self.mu.weight.view(self.out_channels, -1)**2 @ self.unfold(sigma_x).permute(2, 1, 0)).permute(2, 1, 0) + \
            #          (softplus(self.sigma.weight).repeat(1, vec_x.shape[1]) @ (vec_x ** 2).permute(2, 1, 0)).permute(2, 1, 0)
            #sigma_y = sigma_y.view(mu_y.shape[0], mu_y.shape[1], mu_y.shape[2], mu_y.shape[3])
        sigma_y *= 1e-3
        return mu_y, sigma_y

    def kl_term(self):
        if (type(self.kernel_size) == tuple) or (type(self.kernel_size) == list):
            kl = 0.5*torch.mean(self.kernel_size[0] * softplus(self.sigma.weight) + 
                torch.norm(self.mu.weight)**2 - self.kernel_size[0] - self.kernel_size[0] * torch.log(softplus(self.sigma.weight)))
        else:
            kl = 0.5*torch.mean(self.kernel_size * softplus(self.sigma.weight) + 
                torch.norm(self.mu.weight)**2 - self.kernel_size - self.kernel_size * torch.log(softplus(self.sigma.weight)))
        return kl
    


class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1,
                 return_indices=True, ceil_mode=False, tuple_input_flag=False):
        super(MaxPool2d, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.mu = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu_x, sigma_x = mu_x
        mu_y, where = self.mu(mu_x)
        sigma_y = retrieve_elements_from_indices(sigma_x, where)
        return mu_y, sigma_y


class ReLU(torch.nn.Module):
    def __init__(self, tuple_input_flag=False):
        super(ReLU, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.relu = torch.nn.SELU()

    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        mu_a = self.relu(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a
    
class GELU(torch.nn.Module):
    def __init__(self, tuple_input_flag=False):
        super(GELU, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.gelu = torch.nn.GELU()

    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        mu_a = self.gelu(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a
    
class ELU(torch.nn.Module):
    def __init__(self, tuple_input_flag=False):
        super(ELU, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.elu = torch.nn.ELU()

    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        mu_a = self.elu(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a
    
class Sigmoid(torch.nn.Module):
    def __init__(self, tuple_input_flag=False):
        super(Sigmoid, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        mu_a = self.sigmoid(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a
    
    
class TanH(torch.nn.Module):
    def __init__(self, tuple_input_flag=False):
        super(TanH, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.tanh = torch.nn.Tanh()
    
    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        mu_a = self.tanh(mu)
        sigma_a = sigma * (torch.autograd.grad(torch.sum(mu_a), mu, create_graph=True, retain_graph=True)[0]**2)
        return mu_a, sigma_a


class Softmax(torch.nn.Module):
    def __init__(self, dim=1, tuple_input_flag=False, independent_probs=True):
        super(Softmax, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.independent = independent_probs
        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        
        mu_a = self.softmax(mu)
        if self.independent:
            J = mu_a*(1-mu_a)
            sigma = (J**2) * sigma
        else:

            if mu.shape == sigma.shape:
                sigma = torch.diag_embed(sigma)

            pp1 = torch.unsqueeze(mu_a, 2)
            pp2 = torch.unsqueeze(mu_a, 1)
            ppT = pp1 @ pp2
            p_diag = torch.diag_embed(mu_a)
            grad = p_diag - ppT
            sigma = torch.nan_to_num(grad @ (sigma @ grad.mT), 0, 0, 0)

        return mu_a, sigma


class BatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, tuple_input_flag=False):
        super(BatchNorm2d, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.mu = torch.nn.BatchNorm2d(num_features)

    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        mu_a = self.mu(mu)
        bn_param = ((self.mu.weight/(torch.sqrt(self.mu.running_var + self.mu.eps))) ** 2)
        sigma_a = bn_param[None, :, None, None] * sigma
        return mu_a, sigma_a
    

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, tuple_input_flag=False):
        super(LayerNorm, self).__init__()
        self.tuple_input_flag = tuple_input_flag
        self.mu = torch.nn.LayerNorm(normalized_shape)

    def forward(self, mu, sigma=torch.tensor(0., requires_grad=True)):
        if self.tuple_input_flag:
            mu, sigma = mu
        mu_a = self.mu(mu)
        bn_param = ((self.mu.weight.unsqueeze(0)/(torch.sqrt(mu.var((-1), keepdim=True, unbiased=False) + self.mu.eps))) ** 2)
        sigma_a = bn_param * sigma
        return mu_a, sigma_a
    

def ELBOLoss(mu, sigma, y, num_classes=10, criterion='nll'):
    y_hot = torch.nn.functional.one_hot(y, num_classes=num_classes)
    sigma_clamped = torch.log(1+torch.exp(torch.clamp(sigma, 0, 88)))
    if num_classes > 10:
        log_det = torch.mean(torch.log(torch.log(torch.sum(sigma_clamped, dim=1))))
    else:   
        log_det = torch.mean(torch.log(torch.prod(sigma_clamped, dim=1)))
    if criterion == 'nll':
        nll = torch.mean(((y_hot-mu)**2).T @ torch.reciprocal(sigma_clamped))
    elif criterion == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
        nll = criterion(mu, y)
    return log_det, nll


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


def scale_hyperp(log_det, nll, kl):
    # Find the alpha scaling factor
    lli_power = orderOfMagnitude(nll)
    ldi_power = orderOfMagnitude(log_det)
    alpha = 10**(lli_power - ldi_power)     # log_det_i needs to be 1 less power than log_likelihood_i

    beta = list()
    # Find scaling factor for each kl term
    kl = [i.item() for i in kl]
    smallest_power = orderOfMagnitude(min(kl))
    for i in range(len(kl)):
        power = orderOfMagnitude(kl[i])
        power = smallest_power-power
        beta.append(1)
        # beta.append(10.0**power)

    # Find the tau scaling factor
    tau = 10**(smallest_power - lli_power)

    return alpha, tau


def gather_kl(model):
    kl = list()
    for layer in model.modules():
        if hasattr(layer, 'kl_term'):
            kl.append(layer.kl_term())
    return kl


class AdaptiveAvgPool2d(torch.nn.Module):
    def init(self, shape=(1,1)):
        super(AdaptiveAvgPool2d, self).init()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(shape)

    def forward(self, mu, sigma):
        mu_y = self.avgpool(mu)
        sigma_y = self.avgpool(sigma) + torch.var(mu)
        return mu_y, sigma_y




class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, input_flag=False):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = Linear(input_size, 3 * hidden_size, bias=bias, input_flag=input_flag)
        self.h2h = Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.sigmoid = Sigmoid()
        self.tanh = TanH()
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, mu_x, sigma_x=torch.tensor(0., requires_grad=True), mu_hx=None, sigma_hx=None):
        """
        Inputs:
            input: of shape (batch_size, input_size)
            hx: of shape (batch_size, hidden_size)
        Output:
            hy: of shape (batch_size, hidden_size)
        """

        if mu_hx is None:
            mu_hx = torch.autograd.Variable(mu_x.new_zeros(mu_x.size(0), self.hidden_size))
            sigma_hx = torch.autograd.Variable(sigma_x.new_zeros(sigma_x.size(0), self.hidden_size))


        mu_xt, sigma_xt = self.x2h(mu_x, sigma_x)
        mu_ht, sigma_ht = self.h2h(mu_hx, sigma_hx)


        mu_x_reset, mu_x_upd, mu_x_new = mu_xt.chunk(3, 1)
        sigma_x_reset, sigma_x_upd, sigma_x_new = sigma_xt.chunk(3, 1)
        mu_h_reset, mu_h_upd, mu_h_new = mu_ht.chunk(3, 1)
        sigma_h_reset, sigma_h_upd, sigma_h_new = sigma_ht.chunk(3, 1)

        reset_gate_mu, reset_gate_sigma = self.sigmoid(mu_x_reset + mu_h_reset, sigma_x_reset + sigma_h_reset)
        update_gate_mu, update_gate_sigma = self.sigmoid(mu_x_upd + mu_h_upd, sigma_x_upd + sigma_h_upd)

        gated_mu_h_new = reset_gate_mu * mu_h_new
        gated_sigma_h_new = reset_gate_sigma*sigma_h_new + \
                            reset_gate_sigma*torch.square(mu_h_new) + \
                            sigma_h_new*torch.square(reset_gate_mu)
            

        new_gate_mu, new_gate_sigma = self.tanh(mu_x_new + gated_mu_h_new, sigma_x_new + gated_sigma_h_new)

        gated_sigma_hx = update_gate_sigma*sigma_hx + \
                        update_gate_sigma*torch.square(mu_hx) + \
                        sigma_hx*torch.square(update_gate_mu)

        double_gate_sigma = new_gate_sigma*update_gate_sigma + \
                            new_gate_sigma*torch.square(1 - update_gate_mu) + \
                            update_gate_sigma*torch.square(new_gate_mu)

        mu_hy = update_gate_mu * mu_hx + (1 - update_gate_mu) * new_gate_mu

        sigma_hy = gated_sigma_hx + double_gate_sigma

        return mu_hy, sigma_hy