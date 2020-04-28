Navigation Project  ReadMe
******************************************

# [Udacity: Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)  
## Project : Navigation
### Purpose
The purpose of the project is to train an agent to navigate (and collect bananas!) in a large, square world.  
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Environment
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.   
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects  
around the agent's forward direction. Given this information, the agent has to learn how to best select actions.  
Four discrete actions are available, corresponding to:  
0 - move forward.  
1 - move backward.  
2 - turn left.  
3 - turn right.  

### Implementation
**The framework**  
As part of the project requirements I have used [Pytorch](https://pytorch.org/) framework.  

**The Algorithm**  
I have used the [**Double DQN with proportional prioritization**](https://arxiv.org/pdf/1511.05952.pdf) algorithm  
to train the gatent.

**The Network**  
A network with few fully connected layers has been used in my implementation. 
Following is the model  architecture:
Input(state_size) => BatchNorm1d() => Linear(64)=> Dropout(p=0.05) => ReLU() =>  Linear(64) => ReLU() =>  Linear(action_size)  

**The Hyper parameters**  

Parameter | Value | Comment
--- | --- | ---
BUFFER_SIZE |  int(1e5) |  replay buffer size
BATCH_SIZE |  64 |  minibatch size
GAMMA |  0.99 |   discount factor
TAU  |   1e-3  |   for soft update of target parameters
LR |  2e-4 |   learning rate 
UPDATE_EVERY |  4  |  how often to update the network

**The result**
I have challenged myself and have increased the “DONE” criterion from +13 to +16.
The training has been completed within  735 epochs.

The below video compares the performance of the agent before training (random movements) and after training (movements oriented on yellow bananas )
