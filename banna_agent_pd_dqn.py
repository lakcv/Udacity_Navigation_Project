import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 2e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
criterion = nn.MSELoss() #nn.CrossEntropyLoss() #nn.CrossEntropyLoss() # 

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed , pow_alpha = 0, pow_betta = 0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.memory = PreoritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed,alpha=pow_alpha,betta=pow_betta)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # prioritized 
        self.pow_alpha = 0
        self.pow_betta = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                td_errs = self.learn(experiences, GAMMA)
                self.memory.update_priority(td_errs)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device, non_blocking=True)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones ,  siw = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"

        ## Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Get index of argmax qnetwork_target
        _, next_state_actions  = self.qnetwork_local(next_states).max(1, keepdim=True)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_state_actions)
        
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        
        
        #loss = criterion(action_values, target_action_values)
        loss = criterion(siw*Q_expected, siw*Q_targets)
        
        TD_error = (Q_targets - Q_expected).detach().cpu().numpy()
        #loss = F.mse_loss(Q_expected, Q_targets)
        
        #print(loss.data)
        #loss = criterion(torch.clamp(action_values, -1.0, 1.0), torch.clamp(target_action_values, -1.0, 1.0))        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 

        return TD_error
        
                

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device, non_blocking=True)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device, non_blocking=True)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device, non_blocking=True)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device, non_blocking=True)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device, non_blocking=True)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class PreoritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed,alpha=1.0,betta=0.0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) 
        self.priorities = deque(maxlen=buffer_size)          
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priority = namedtuple("Priorities", field_names=["priority"])
        self.seed = random.seed(seed)
        self.p_max = 1.0
        self.alpha = alpha
        self.betta = betta
        self.last_selected_indexs = None
        self.buffer_size = buffer_size
        self.max_siw = None
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(e)
        p = self.priority(self.p_max)
        self.priorities.append(p)
    
    def sample(self):

        """Randomly sample a batch of experiences from memory."""
        """ The sampling distribution is a function of transition priority"""
        """ 
            PRIORITIZED EXPERIENCE REPLAY
            Tom Schaul, John Quan, Ioannis Antonoglou and David Silver
            Google DeepMind        
            {fschaul,johnquan,ioannisa,davidsilverg}@google.com
        """
        priorities =np.vstack([m.priority**self.alpha for m in self.priorities if m is not None])
        prb_dist = priorities / sum(priorities)
        if  self.max_siw is None:
            sample_importance_weights = (prb_dist * self.buffer_size)**(-self.betta)
            sample_importance_weights /= np.max(sample_importance_weights)
            self.max_siw = 1.0
        else:
            sample_importance_weights = (prb_dist * self.buffer_size)**(-self.betta)/self.max_siw
            max_siw_temp = np.max(sample_importance_weights)
            if max_siw_temp > self.max_siw:
                self.max_siw = max_siw_temp
                
        indices = [i for i in range(len(priorities))]
        self.last_selected_indexs = random.choices(indices, weights=prb_dist, k=self.batch_size)
        
        """Update sample a batch of experiences from memory."""
        
        experiences = [self.memory[i] for i in self.last_selected_indexs]
        siw_np = [sample_importance_weights[i] for i in self.last_selected_indexs]
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device, non_blocking=True)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device, non_blocking=True)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device, non_blocking=True)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device, non_blocking=True)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device, non_blocking=True)
        siw = torch.from_numpy(np.vstack(siw_np)).float().to(device, non_blocking=True)
        return (states, actions, rewards, next_states, dones,siw)
        
    def update_priority(self, td_errors):
        for indx,td_err in zip(self.last_selected_indexs,list(td_errors)):
            self.priorities[indx] = self.priorities[indx]._replace(priority=(np.abs(td_err)+0.00001))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)