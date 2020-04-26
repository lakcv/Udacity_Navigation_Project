# [Udacity: Deep Reinforsment Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)  
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

I have used the [**Double DQN with proportional prioritization**](https://arxiv.org/pdf/1511.05952.pdf) algorithm  
in the train
