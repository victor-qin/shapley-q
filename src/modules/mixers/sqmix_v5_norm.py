import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ShapleyQMixer(nn.Module):
    def __init__(self, args):
        super(ShapleyQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.arch = args.arch
        self.embed_dim = args.mixing_embed_dim
        self.n_actions = args.n_actions

        self.sample_size = args.sample_size
        self.state_normaliser = RunningMeanStd(self.state_dim)

        # w(s,u)
        if self.arch == "observation_action":
            # w,f,g takes [state, u] as input
            w_input_size = self.state_dim + 2*self.n_actions
            # print (f"This is the w_input_size: {w_input_size}")
        else:
            raise Exception("{} is not a valid ShapleyQ architecture".format(self.arch))

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * 2)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * 2))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                            nn.ReLU(),
                            nn.Linear(self.embed_dim, 1))

    def sample_grandcoalitions(self, batch_size):
        """
        E.g. batch_size = 2, n_agents = 3:

        >>> grand_coalitions
        tensor([[2, 0, 1],
                [1, 2, 0]])

        >>> subcoalition_map
        tensor([[[[1., 1., 1.],
                [1., 0., 0.],
                [1., 1., 0.]]],

                [[[1., 1., 0.],
                [1., 1., 1.],
                [1., 0., 0.]]]])

        >>> individual_map
        tensor([[[[0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.]]],

                [[[0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]]]])
        """
        seq_set = th.tril(th.ones(self.n_agents, self.n_agents).cuda(), diagonal=0, out=None)
        grand_coalitions = th.multinomial(th.ones(batch_size*self.sample_size, 
                                          self.n_agents).cuda()/self.n_agents, 
                                          self.n_agents, 
                                          replacement=False)
        individual_map = th.zeros(batch_size*self.sample_size*self.n_agents, self.n_agents).cuda()
        individual_map.scatter_(1, grand_coalitions.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_agents, self.n_agents)
        subcoalition_map = th.matmul(individual_map, seq_set)
        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, 
                                                                self.n_agents, 
                                                                self.n_agents).contiguous().view(batch_size, 
                                                                                                 self.sample_size, 
                                                                                                 self.n_agents, 
                                                                                                 self.n_agents) # shape = (b, n_s, n, n)
        return subcoalition_map, individual_map, grand_coalitions

    def get_w_estimate(self, states, agent_qs):
        batch_size = states.size(0)

        # get subcoalition map including agent i
        subcoalition_map, individual_map, grand_coalitions = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)

        # reshape the grand coalition map for rearranging the sequence of actions of agents
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 1) # shape = (b, n_s, n, n, 1)

        # remove agent i from the subcloation map
        subcoalition_map_no_i = subcoalition_map - individual_map
        subcoalition_map_no_i = subcoalition_map_no_i.unsqueeze(-1).expand(batch_size, 
                                                                 self.sample_size, 
                                                                 self.n_agents, 
                                                                 self.n_agents, 
                                                                 1) # shape = (b, n_s, n, n, 1)
        
        # reshape actions for further process on coalitions
        reshape_agent_qs = agent_qs.unsqueeze(1).unsqueeze(2).expand(batch_size, 
                                                        self.sample_size, 
                                                        self.n_agents, 
                                                        self.n_agents, 
                                                        1).gather(3, grand_coalitions) # shape = (b, n, 1) -> (b, 1, 1, n, 1) -> (b, n_s, n, n, 1)

        # get actions of its coalition memebers for each agent
        agent_qs_coalition = reshape_agent_qs * subcoalition_map_no_i # shape = (b, n_s, n, n, 1)

        # get actions vector of its coalition members for each agent
        # agent_qs_coalition_norm_vec = agent_qs_coalition.mean(dim=-2) * subcoalition_map_no_i.sum(dim=-2) # shape = (b, n_s, n, 1)
        subcoalition_map_no_i_ = subcoalition_map_no_i.sum(dim=-2).clone()
        subcoalition_map_no_i_[subcoalition_map_no_i.sum(dim=-2)==0] = 1
        agent_qs_coalition_norm_vec = agent_qs_coalition.sum(dim=-2) / subcoalition_map_no_i_ # shape = (b, n_s, n, 1)

        # normalise among the sample_size
        # agent_qs_coalition_norm_vec = agent_qs_coalition_norm_vec.mean(dim=1) # shape = (b, n, 1)

        # get action vector of each agent
        agent_qs_individual = agent_qs.unsqueeze(1).expand_as(agent_qs_coalition_norm_vec) # shape = (b, n_s, n, 1)

        reshape_agent_qs_coalition_norm_vec = agent_qs_coalition_norm_vec.contiguous().view(-1, 1) # shape = (b*n_s*n, 1)
        reshape_agent_qs_individual = agent_qs_individual.contiguous().view(-1, 1) # shape = (b*n_s*n, 1)
        reshape_states = states.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_agents, self.state_dim).contiguous().view(-1, self.state_dim) # shape = (b*n_s*n, s)

        # print (f"This is the reshape_actions_coalition_norm_vec: {reshape_actions_coalition_norm_vec.size()}")
        # print (f"This is the reshape_actions_individual: {reshape_actions_individual.size()}")
        # print (f"This is the reshape_states: {reshape_states.size()}")

        inputs = th.cat([reshape_agent_qs_coalition_norm_vec, reshape_agent_qs_individual], dim=-1).unsqueeze(1) # shape = (b*n_s*n, 1, 2*1)

        # First layer
        w1 = th.abs(self.hyper_w_1(reshape_states))
        b1 = self.hyper_b_1(reshape_states)
        w1 = w1.view(-1, 2, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(inputs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(reshape_states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(reshape_states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        w_estimates = th.abs(y).view(batch_size, self.sample_size, self.n_agents) # shape = (b, n_s, n)
        # normalise among the sample_size
        w_estimates = w_estimates.mean(dim=1) # shape = (b, n)

        return w_estimates

    def forward(self, states, actions, agent_qs, max_filter, target=True):
        # agent_qs, max_filter = (b, t, n)
        reshape_states = states.contiguous().view(-1, self.state_dim)
        self.state_normaliser.update(reshape_states)
        # normalise state
        reshape_states = (reshape_states - self.state_normaliser.mean) / th.sqrt(self.state_normaliser.var)
        reshape_agent_qs = agent_qs.unsqueeze(-1).contiguous().view(-1, self.n_agents, 1)
        if target:
            return th.sum(agent_qs, dim=2, keepdim=True)
        else:
            w_estimates = self.get_w_estimate(reshape_states, reshape_agent_qs)
            # restrict the range of w to (1, 2)
            w_estimates = w_estimates + 1
            w_estimates = w_estimates.contiguous().view(states.size(0), states.size(1), self.n_agents)
            if max_filter is None:
                return (w_estimates * agent_qs).sum(dim=2, keepdim=True), w_estimates
            else:
                # agent with non-max action will be given 1
                non_max_filter = 1 - max_filter
                # if the agent with the max-action then w=1
                # otherwise the agent will use the learned w
                return ( (w_estimates * non_max_filter + max_filter) * agent_qs).sum(dim=2, keepdim=True), w_estimates


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        super(RunningMeanStd, self).__init__()
        self.mean = th.zeros(shape).cuda()
        self.var = th.ones(shape).cuda()
        self.count = epsilon

    def update(self, x):
        batch_mean = th.mean(x, dim=0)
        batch_var = th.var(x, dim=0)
        batch_count = x.size(0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count