import torch
import torch.nn as nn
from dreamerv2.utils.rssm_vdp import RSSMUtils, RSSMContState, RSSMDiscState
import dreamerv2.models.vdp as vdp

class RSSM(nn.Module, RSSMUtils):
    def __init__(
        self,
        action_size,
        rssm_node_size,
        embedding_size,
        device,
        rssm_type,
        info,
        act_fn=vdp.ELU,  
    ):
        nn.Module.__init__(self)
        RSSMUtils.__init__(self, rssm_type=rssm_type, info=info)
        self.device = device
        self.action_size = action_size
        self.node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.act_fn = act_fn
        self.rnn = vdp.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()
    
    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action 
        and embed it to deter size for rnn input
        """
        # this needs an input flag because the inputs are categorical one-hots, not gaussians
        fc_embed_state_action = [vdp.Linear(self.stoch_size + self.action_size, self.deter_size, input_flag=True)]
        fc_embed_state_action += [self.act_fn(tuple_input_flag=True)]
        return nn.Sequential(*fc_embed_state_action)
    
    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state 
        and output prior over stochastic state
        """
        temporal_prior = [vdp.Linear(self.deter_size, self.node_size, tuple_input_flag=True)]
        temporal_prior += [self.act_fn(tuple_input_flag=True)]
        temporal_prior += [vdp.Linear(self.node_size, self.stoch_size, tuple_input_flag=True)]
        """
        this if statement is no longer needed, because we don't need to regress mean and variance
        for continuous priors, we just get them the bayes way

        if self.rssm_type == 'discrete':
            temporal_prior += [vdp.Linear(self.node_size, self.stoch_size, tuple_input_flag=True)]
        elif self.rssm_type == 'continuous':
            temporal_prior += [vdp.Linear(self.node_size, 2 * self.stoch_size, tuple_input_flag=True)]"""
        return nn.Sequential(*temporal_prior)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state 
        and output posterior over stochastic states
        """
        temporal_posterior = [vdp.Linear(self.deter_size + self.embedding_size, self.node_size, tuple_input_flag=True)]
        temporal_posterior += [self.act_fn(tuple_input_flag=True)]
        temporal_posterior += [vdp.Linear(self.node_size, self.stoch_size, tuple_input_flag=True)]
        """
        this if statement is no longer needed, because we don't need to regress mean and variance
        for continuous priors, we just get them the bayes way

        if self.rssm_type == 'discrete':
            temporal_prior += [vdp.Linear(self.node_size, self.stoch_size, tuple_input_flag=True)]
        elif self.rssm_type == 'continuous':
            temporal_prior += [vdp.Linear(self.node_size, 2 * self.stoch_size, tuple_input_flag=True)]"""
        return nn.Sequential(*temporal_posterior)
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        # embed state action has an input flag, so only input mu's
        state_action_embed_mu, state_action_embed_sigma = self.fc_embed_state_action(
                            torch.cat([prev_rssm_state.stoch*nonterms, prev_action], dim=-1)) 
        
        # rnn makes gaussian "deterministic" states
        deter_state_mu, deter_state_sigma = self.rnn(state_action_embed_mu, 
                                            state_action_embed_sigma, 
                                            prev_rssm_state.deter_mu*nonterms,
                                            prev_rssm_state.deter_sigma*nonterms)
        
        if self.rssm_type == 'discrete':
            prior_mean, prior_std = self.fc_prior((deter_state_mu, deter_state_sigma))
            stats = {'mean':prior_mean, 'std':vdp.i_softplus(prior_std)}
            prior_stoch_state = self.get_stoch_state(stats)
            prior_rssm_state = RSSMDiscState(prior_mean, prior_std, prior_stoch_state, deter_state_mu, deter_state_sigma)

        elif self.rssm_type == 'continuous':
            # no more regression, get these the bayes way
            #prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            prior_mean, prior_std = self.fc_prior(deter_state_mu, deter_state_sigma)
            # needs i_softplus because get_stoch_state does softplus and I'm trying to minimize duplicate files
            stats = {'mean':prior_mean, 'std':prior_std}
            prior_stoch_state, std = self.get_stoch_state(stats)
            prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state_mu, deter_state_sigma)
        return prior_rssm_state

    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        for t in range(horizon):
            model_state_mu, model_state_sigma = self.get_model_state(rssm_state)
            action, action_dist = actor(((model_state_mu).detach(), (model_state_sigma).detach()))
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy

    def rssm_observe(self, obs_embed_mu, obs_embed_sigma, prev_action, prev_nonterm, prev_rssm_state):
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state_mu = prior_rssm_state.deter_mu 
        deter_state_sigma = prior_rssm_state.deter_sigma
        mu_x = torch.cat([deter_state_mu, obs_embed_mu], dim=-1)
        sigma_x = torch.cat([deter_state_sigma, obs_embed_sigma], dim=-1)
        if self.rssm_type == 'discrete':
            posterior_mean, posterior_std = self.fc_posterior((mu_x, sigma_x))
            stats = {'mean':posterior_mean, 'std':posterior_std}
            posterior_stoch_state = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMDiscState(posterior_mean, posterior_std, posterior_stoch_state, 
                                                 deter_state_mu, deter_state_sigma)
        
        elif self.rssm_type == 'continuous':
            # no more regression, get these the bayes way
            #posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            posterior_mean, posterior_std = self.fc_posterior(mu_x, sigma_x)
            # needs i_softplus because get_stoch_state does softplus and I'm trying to minimize duplicate files
            stats = {'mean':posterior_mean, 'std':posterior_std}
            posterior_stoch_state, std = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, 
                                                 deter_state_mu, deter_state_sigma)
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, seq_len:int, obs_embed_mu: torch.Tensor, obs_embed_sigma: torch.Tensor, 
                            action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state):
        priors = []
        posteriors = []
        for t in range(seq_len):
            prev_action = action[t]*nonterms[t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed_mu[t], obs_embed_sigma[t], 
                                                                       prev_action, nonterms[t], prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=0)
        post = self.rssm_stack_states(posteriors, dim=0)
        return prior, post
        