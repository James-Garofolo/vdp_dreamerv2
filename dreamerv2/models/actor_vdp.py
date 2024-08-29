import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import dreamerv2.models.vdp as vdp

class DiscreteActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info,
        exp_scaler=20
    ):
        super().__init__()
        self.explore_buffer = []
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.exp_scaler = exp_scaler
        self.model = self._build_model()

    def _build_model(self):
        model = [vdp.Linear(self.deter_size + self.stoch_size, self.node_size, tuple_input_flag=True)]
        model += [self.act_fn(tuple_input_flag=True)]
        for i in range(1, self.layers):
            model += [vdp.Linear(self.node_size, self.node_size, tuple_input_flag=True)]
            model += [self.act_fn(tuple_input_flag=True)]

        if self.dist == 'one_hot':
            model += [vdp.Linear(self.node_size, self.action_size, output_flag = True, tuple_input_flag=True, output_scaler=self.exp_scaler)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model) 

    def forward(self, model_state, explore=False):
        action_dist, sigmas = self.get_action_dist(model_state, explore)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mean
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist, sigmas

    def get_action_dist(self, modelstate, explore=False):
        mus, sigmas = self.model(modelstate)
        if explore:
            logits = mus + torch.sqrt(sigmas)*torch.randn_like(sigmas)
            self.explore_buffer.append((torch.argmax(logits)!=torch.argmax(mus)).item())
            if len(self.explore_buffer) > 1000:
                self.explore_buffer.pop(0)
        else:
            logits = mus
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits), sigmas        
        else:
            raise NotImplementedError
            
    """def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action

        raise NotImplementedError"""