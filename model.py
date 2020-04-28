import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        ''' 
            my model : in(state_size)::bnorm ::fc1(32)::drop(0.2)::fc2(64)::fc2(64) ::drop(0.1)::fc2(action_size)
        '''
        self.hidden_dims = [64 , 64]
        self.state_size  =  state_size
        self.action_size = action_size
        self.main = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, self.hidden_dims[0]),
            nn.Dropout(p=0.05), #20 % probability 
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            #nn.Dropout(p=0.1), #10% probability
            nn.Linear(self.hidden_dims[1], self.action_size)
        )
            

    def forward(self, state):
        """Build a network that maps state -> action values."""
        '''logits = self.model.apply(state)
        return F.softmax(logits)'''
        return self.main(state)
        
