from collections import namedtuple
import torch.distributions as td
import torch
import torch.nn.functional as F
from typing import Union
from dreamerv2.models import vdp

RSSMDiscState = namedtuple('RSSMDiscState', ['logit_mean', 'logit_std', 'stoch', 'deter_mu', 'deter_sigma'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter_mu', 'deter_sigma'])  

RSSMState = Union[RSSMDiscState, RSSMContState]

class RSSMUtils(object):
    '''utility functions for dealing with rssm states'''
    def __init__(self, rssm_type, info):
        self.vdp_softmax = vdp.Softmax()
        self.rssm_type = rssm_type
        if rssm_type == 'continuous':
            self.deter_size = info['deter_size']
            self.stoch_size = info['stoch_size']
            self.min_std = info['min_std']
        elif rssm_type == 'discrete':
            self.deter_size = info['deter_size']
            self.class_size = info['class_size']
            self.category_size = info['category_size']
            self.stoch_size  = self.class_size*self.category_size
        else:
            raise NotImplementedError

    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                seq_to_batch(rssm_state.logit_mean[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.logit_std[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter_mu[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter_sigma[:seq_len], batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                seq_to_batch(rssm_state.mean[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.std[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter_mu[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter_sigma[:seq_len], batch_size, seq_len)
            )
        
    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                batch_to_seq(rssm_state.logit_mean, batch_size, seq_len),
                batch_to_seq(rssm_state.logit_std, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter_mu, batch_size, seq_len),
                batch_to_seq(rssm_state.deter_sigma, batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                batch_to_seq(rssm_state.mean, batch_size, seq_len),
                batch_to_seq(rssm_state.std, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter_mu, batch_size, seq_len),
                batch_to_seq(rssm_state.deter_sigma, batch_size, seq_len)
            )
        
    def get_dist(self, rssm_state):
        if self.rssm_type == 'discrete':
            shape = rssm_state.logit_mean.shape
            logit_mu = torch.reshape(rssm_state.logit_mean, shape = (*shape[:-1], self.category_size, self.class_size))
            logit_std = torch.reshape(rssm_state.logit_std, shape = (*shape[:-1], self.category_size, self.class_size))
            logit = logit_mu + torch.sqrt(logit_std) * torch.rand_like(logit_std)
            return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
        elif self.rssm_type == 'continuous':
            return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)

    def get_stoch_state(self, stats):
        if self.rssm_type == 'discrete':
            logit_mu = stats['mean']
            logit_sigma = stats['std']
            shape = logit_mu.shape
            logit_mu = torch.reshape(logit_mu, shape = (*shape[:-1], self.category_size, self.class_size))
            logit_sigma = torch.reshape(logit_sigma, shape = (*shape[:-1], self.category_size, self.class_size))
            logit = logit_mu + torch.sqrt(logit_sigma) * torch.rand_like(logit_sigma)
            dist = torch.distributions.OneHotCategorical(logits=logit)        
            stoch = dist.sample()
            stoch += dist.probs - dist.probs.detach()
            return torch.flatten(stoch, start_dim=-2, end_dim=-1)

        elif self.rssm_type == 'continuous':
            mean = stats['mean']
            std = stats['std']
            std = torch.sqrt(std) + self.min_std
            return mean + std*torch.randn_like(mean), std

    def rssm_stack_states(self, rssm_states, dim):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                torch.stack([state.logit_mean for state in rssm_states], dim=dim),
                torch.stack([state.logit_std for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter_mu for state in rssm_states], dim=dim),
                torch.stack([state.deter_sigma for state in rssm_states], dim=dim)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                torch.stack([state.mean for state in rssm_states], dim=dim),
                torch.stack([state.std for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter_mu for state in rssm_states], dim=dim),
                torch.stack([state.deter_sigma for state in rssm_states], dim=dim)
            )

    def get_model_state(self, rssm_state):
        if self.rssm_type == 'discrete':
            _, stoch_probs_sigma = self.vdp_softmax(rssm_state.logit_mean,
                                                                 rssm_state.logit_std)
            return torch.cat((rssm_state.deter_mu, 
                            rssm_state.stoch), dim=-1), \
                    torch.cat((rssm_state.deter_sigma, 
                            rssm_state.logit_std), dim=-1)
                            #stoch_probs_sigma), dim=-1)
        
        elif self.rssm_type == 'continuous':
            return torch.cat((rssm_state.deter_mu, rssm_state.stoch), dim=-1),\
                   torch.cat((rssm_state.deter_sigma, rssm_state.std))

    def rssm_detach(self, rssm_state):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                rssm_state.logit_mean.detach(),  
                rssm_state.logit_std.detach(), 
                rssm_state.stoch.detach(),
                rssm_state.deter_mu.detach(),
                rssm_state.deter_sigma.detach()
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                rssm_state.mean.detach(),
                rssm_state.std.detach(),  
                rssm_state.stoch.detach(),
                rssm_state.deter_mu.detach(),
                rssm_state.deter_sigma.detach()
            )

    def _init_rssm_state(self, batch_size, **kwargs):
        if self.rssm_type  == 'discrete':
            return RSSMDiscState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device)
            )
            
def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
    return seq_data

