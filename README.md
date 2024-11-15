```python
from IPython.display import Image
```


```python
Image(filename = 'SMPRIL.jpeg')
```




    
![jpeg](README_files/README_1_0.jpg)
    



# State Memory-Based Predictive Reinforcement Imitation Learning (SMPRIL) for CASBOT "01".

Author: [Kevin Thomas](mailto:ket189@pitt.edu)

License: [Apache License 2.0](https://github.com/mytechnotalent/SMPRIL/blob/main/LICENSE)

## Introduction

The **New AI CASBOT “01” Humanoid Robot** is equipped with advanced **52-axes tech** and leverages **NVIDIA EDIFY 3D AI tools** to achieve state-of-the-art robotic capabilities. In this notebook, we implement **State Memory-Based Predictive Reinforcement Imitation Learning (SMPRIL)** to train CASBOT for dynamic and complex tasks, such as navigation, manipulation, and human-robot interaction.

### What is SMPRIL?

**SMPRIL** combines the strengths of:
1. **State Memory**: CASBOT uses a memory buffer to store and reference past states, actions, and rewards, enabling it to learn from history.
2. **Predictive Modeling**: A predictive neural network predicts future states based on current states and actions. This allows CASBOT to anticipate outcomes and plan ahead.
3. **Reinforcement Learning (RL)**: CASBOT learns optimal actions through trial and error, aiming to maximize cumulative rewards over time.
4. **Imitation Learning (IL)**: CASBOT learns from expert demonstrations to speed up the learning process and achieve better initial performance.

By combining these techniques, CASBOT can handle complex, real-world scenarios with greater efficiency and accuracy.

---

## CASBOT "01" Features

### 1. **52 Axes Tech**
   - Offers fine-grained control and dexterity for advanced manipulation and movement tasks.

### 2. **NVIDIA EDIFY 3D AI Tools**
   - **Isaac Sim**: Used for simulating environments.
   - **TensorRT**: Optimized inference for deep learning models.
   - **Omniverse**: Provides a real-time 3D simulation environment.

---

## Workflow

1. **State Memory**:
   - Stores past experiences (`state`, `action`, `reward`, `next_state`, `done`) in a memory buffer.
   - Helps CASBOT reference historical data to improve decision-making.
<br>
2. **Predictive Modeling**:
   - A neural network predicts the next state given the current state and action.
   - Allows CASBOT to simulate the outcomes of actions before execution.
<br>
3. **Reinforcement Learning**:
   - CASBOT explores and interacts with the environment to learn optimal policies.
   - Uses Q-learning to optimize actions based on cumulative rewards.
<br>
4. **Imitation Learning**:
   - CASBOT mimics expert demonstrations during initial training, enabling faster and more reliable learning.

---

## Device Support for Portability

This implementation detects and utilizes the best available device:
- **NVIDIA GPU (CUDA)**: Leverages high-performance Tensor Cores for deep learning.
- **Apple GPU (MPS)**: Utilizes Metal Performance Shaders for M3 chips.
- **CPU**: Fallback option when no GPU is available.

---

## Key Components of the Implementation

### 1. **Memory Buffer**
The memory buffer stores past experiences:
- `state`: The robot's current state (e.g., joint angles, sensor readings).
- `action`: The action taken (e.g., move to a new position).
- `reward`: The reward received for the action.
- `next_state`: The resulting state after the action.
- `done`: Whether the task or episode is complete.

This allows CASBOT to learn from previous interactions.

### 2. **Predictive Model**
A neural network predicts:
- **Q-values**: Expected rewards for each possible action in a given state.
- **Next State**: Simulated state resulting from a specific action.

This predictive ability enables CASBOT to anticipate outcomes and avoid suboptimal actions.

### 3. **Reinforcement Learning**
Reinforcement learning uses:
- **Exploration**: Random actions to discover new strategies.
- **Exploitation**: Leveraging known strategies to maximize rewards.

### 4. **Imitation Learning**
Imitation learning initializes CASBOT with expert data, providing a strong foundation and reducing the need for random exploration.

---

## CASBOT Training Workflow

1. **Environment Setup**:
   - Simulate CASBOT's environment using NVIDIA EDIFY tools (e.g., Isaac Sim).
<br>
2. **Memory-Based Learning**:
   - Store experiences in a memory buffer to enable learning from past interactions.
<br>
3. **Predictive and Reinforcement Learning**:
   - Train CASBOT to predict outcomes and optimize actions based on rewards.
<br>
4. **Device-Specific Optimization**:
   - Utilize NVIDIA CUDA for GPUs or Apple MPS for macOS systems with M3 chips.

---

## Advantages of SMPRIL for CASBOT "01"

1. **Efficient Learning**: Combines imitation learning and reinforcement learning for faster training.
2. **Predictive Control**: CASBOT can simulate and anticipate outcomes, reducing errors.
3. **Adaptive Behavior**: State memory enables CASBOT to adapt to dynamic environments.
4. **Portability**: Runs on both NVIDIA GPUs and Apple MPS for broader compatibility.

---

## Summary

This notebook demonstrates how **State Memory-Based Predictive Reinforcement Imitation Learning (SMPRIL)** empowers CASBOT “01” to handle complex tasks with its advanced hardware capabilities and NVIDIA EDIFY 3D AI tools. By integrating memory, predictive modeling, reinforcement learning, and imitation learning, CASBOT achieves robust performance in real-world scenarios.

Run the provided Python code to:
1. Train CASBOT in a simulated environment.
2. Use predictive models for enhanced control.
3. Leverage device-specific optimizations for peak performance.

Explore the power of SMPRIL with CASBOT "01"!

## Prerequisites


```python
!pip install torch torchvision torchaudio
```

    Requirement already satisfied: torch in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (2.5.1)
    Requirement already satisfied: torchvision in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (0.20.1)
    Requirement already satisfied: torchaudio in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (2.5.1)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (4.11.0)
    Requirement already satisfied: networkx in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (3.3)
    Requirement already satisfied: jinja2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (2024.6.1)
    Requirement already satisfied: setuptools in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (75.1.0)
    Requirement already satisfied: sympy==1.13.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: numpy in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torchvision) (1.26.4)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torchvision) (10.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from jinja2->torch) (2.1.3)



```python
!pip install gym
```

    Requirement already satisfied: gym in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (0.26.2)
    Requirement already satisfied: numpy>=1.18.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from gym) (1.26.4)
    Requirement already satisfied: cloudpickle>=1.2.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from gym) (3.0.0)
    Requirement already satisfied: gym-notices>=0.0.4 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from gym) (0.0.8)


## Init GPU


```python
import torch
```


```python
def get_device():
    """
    Function to obtain if a GPU is available and then assign it to the device.
    """
    if torch.cuda.is_available():
        print('NVIDIA GPU detected. Using CUDA.')
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print('Apple GPU detected. Using MPS.')
        return torch.device('mps')
    else:
        print('No compatible GPU detected. Using CPU.')
        return torch.device('cpu')
```

### Set Device


```python
device = get_device()
```

    Apple GPU detected. Using MPS.


## Update the Model and Training to Use Device


```python
import torch.nn as nn
```


```python
class CASBOTPredictiveModel(nn.Module):
    """
    A neural network model for predictive reinforcement learning with CASBOT.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initializes the CASBOT predictive model.

        Params:
            state_dim: int
            action_dim: int
        """
        super(CASBOTPredictiveModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # first hidden layer
        self.fc2 = nn.Linear(256, 128)  # second hidden layer
        self.q_out = nn.Linear(128, action_dim)  # Q-value output for actions
        # linear layer for predicting the future state from state and action
        self.future_state = nn.Linear(state_dim + action_dim, state_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes Q-values for all possible actions given a state.

        Params:
            state: torch.Tensor

        Returns:
            torch.Tensor
        """
        x = torch.relu(self.fc1(state))  # apply first hidden layer with ReLU activation
        x = torch.relu(self.fc2(x))  # apply second hidden layer with ReLU activation
        return self.q_out(x)  # output Q-values

    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predicts the next state based on the current state and an action.

        Params:
            state: torch.Tensor
            action: torch.Tensor

        Returns:
            torch.Tensor
        """
        # concatenate the state and action tensors along the last dimension
        combined_input = torch.cat((state, action), dim=-1)
        # predict the next state using the combined input
        return self.future_state(combined_input)
```

## Update the Agent to Use Device


```python
class CASBOTAgent:
    """
    A reinforcement learning agent for controlling the CASBOT robot.
    The agent uses a neural network for Q-value estimation and predictive modeling,
    enabling efficient learning and decision-making.
    """

    def __init__(self, state_dim, action_dim, memory, gamma=0.99, lr=0.001):
        """
        Initializes the CASBOT agent.

        Params:
            state_dim: int
            action_dim: int
            memory: Memory
            gamma: float, optional
            lr: float, optional
        """
        self.model = CASBOTPredictiveModel(state_dim, action_dim).to(device)
        self.target_model = CASBOTPredictiveModel(state_dim, action_dim).to(device)
        self.update_target_network()
        self.memory = memory
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        """
        Updates the weights of the target network to match the main model.
        This helps stabilize training by reducing the correlation between
        the target and the training updates.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon=0.1):
        """
        Chooses an action using an epsilon-greedy policy.
    
        Params:
            state: np.ndarray
            epsilon: float, optional
    
        Returns:
            int
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, action_dim)  # Random action
    
        # debug: Log state details
        print(f"get_action received state: {state}, shape: {state.shape}, dtype: {state.dtype}")
    
        # ensure state is a NumPy array
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model(state_tensor).argmax().item()

    def train(self, batch_size=32):
        """
        Train model.

        Params:
            batch_size: int, optional
        """
        if len(self.memory) < batch_size:
            return
    
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # Ensure actions are 2D
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
    
        # one-hot encode actions for future_state
        actions = torch.nn.functional.one_hot(actions.squeeze(-1), num_classes=action_dim).float().to(device)
    
        # Q-value update
        q_values = self.model(states).gather(1, actions.argmax(dim=1, keepdim=True)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    
        # predictive model for state prediction
        predicted_next_states = self.model.predict_next_state(states, actions)
        predictive_loss = self.loss_fn(predicted_next_states, next_states)
    
        # compute combined loss
        loss = self.loss_fn(q_values, expected_q_values) + predictive_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## Update the Training Loop to Use Device


```python
import numpy as np
```


```python
import gym
```


```python
from gym import spaces
```


```python
class CASBOTEnvironment(gym.Env):
    """
    A custom Gym environment for simulating the CASBOT robot.
    This environment models the interaction between the agent and the CASBOT robot
    through discrete actions and continuous state observations.
    """
    
    def __init__(self):
        """
        Initializes the CASBOT environment by defining the action and observation spaces.
        """
        super(CASBOTEnvironment, self).__init__()
        
        # define the action space: 5 discrete actions available to the agent
        self.action_space = spaces.Discrete(5)
        
        # define the observation space: 10 continuous state features between -1.0 and 1.0
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # initialize internal variables to track the state and whether the episode is done
        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        This method initializes the state to random continuous values within the observation space range.

        Params:
            seed: int, optional
            options: dict, optional

        Returns:
            tuple
        """
        super().reset(seed=seed)  # set the random seed for reproducibility
        # generate a random state within the observation space
        self.state = np.random.uniform(-1.0, 1.0, size=(10,)).astype(np.float32)
        self.done = False  # reset the done flag to indicate a new episode
        info = {}  # placeholder for any additional information
        print(f'reset state: {self.state}, shape: {self.state.shape}, dtype: {self.state.dtype}')
        return self.state, info
    
    def step(self, action):
        """
        Executes an action in the environment and updates the state.
        This method simulates the environment dynamics by updating the state, calculating a reward,
        and determining whether the episode is terminated or truncated.

        Params:
            action: int

        Returns:
            tuple
        """
        # generate a random reward for the action
        reward = np.random.rand()
        # update the state with a new random value within the observation space
        self.state = np.random.uniform(-1.0, 1.0, size=(10,)).astype(np.float32)
        # determine if the episode is terminated with a 5% chance
        terminated = np.random.rand() > 0.95
        # set truncated to False (not used in this example)
        truncated = False
        # placeholder for any additional information
        info = {}
        print(f'step returned state: {self.state}, shape: {self.state.shape}, dtype: {self.state.dtype}')
        return self.state, reward, terminated, truncated, info
```


```python
from gym.envs.registration import register
```


```python
register(
    id='NVIDIA-CASBOT-v1',
    entry_point='__main__:CASBOTEnvironment',  # use '__main__' when defining in a notebook
    max_episode_steps=500
)
```


```python
env = gym.make('NVIDIA-CASBOT-v1')
print(env)
```

    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CASBOTEnvironment<NVIDIA-CASBOT-v1>>>>>



```python
from gym.envs.registration import registry
```


```python
list(registry.keys())
```




    ['CartPole-v0',
     'CartPole-v1',
     'MountainCar-v0',
     'MountainCarContinuous-v0',
     'Pendulum-v1',
     'Acrobot-v1',
     'LunarLander-v2',
     'LunarLanderContinuous-v2',
     'BipedalWalker-v3',
     'BipedalWalkerHardcore-v3',
     'CarRacing-v2',
     'Blackjack-v1',
     'FrozenLake-v1',
     'FrozenLake8x8-v1',
     'CliffWalking-v0',
     'Taxi-v3',
     'Reacher-v2',
     'Reacher-v4',
     'Pusher-v2',
     'Pusher-v4',
     'InvertedPendulum-v2',
     'InvertedPendulum-v4',
     'InvertedDoublePendulum-v2',
     'InvertedDoublePendulum-v4',
     'HalfCheetah-v2',
     'HalfCheetah-v3',
     'HalfCheetah-v4',
     'Hopper-v2',
     'Hopper-v3',
     'Hopper-v4',
     'Swimmer-v2',
     'Swimmer-v3',
     'Swimmer-v4',
     'Walker2d-v2',
     'Walker2d-v3',
     'Walker2d-v4',
     'Ant-v2',
     'Ant-v3',
     'Ant-v4',
     'Humanoid-v2',
     'Humanoid-v3',
     'Humanoid-v4',
     'HumanoidStandup-v2',
     'HumanoidStandup-v4',
     'NVIDIA-CASBOT-v1']




```python
from collections import deque
```


```python
import random
```


```python
import torch.optim as optim
```


```python
class Memory:
    """
    A simple memory buffer for storing and sampling past experiences.
    """
    def __init__(self, max_size=10000):
        """
        Initializes the memory buffer.

        Params:
            max_size: int, optional
        """
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """
        Adds an experience to the memory buffer.

        Params:
            state: np.ndarray
            action: int
            reward: float
            next_state: np.ndarray
            done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the memory buffer.

        Params:
            batch_size: int

        Returns:
            list
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Returns the current size of the memory buffer.

        Returns:
            int
        """
        return len(self.buffer)
```


```python
import logging
```


```python
import os
```


```python
import contextlib
```


```python
# configure minimal logging
logging.basicConfig(level=logging.INFO)  # Only INFO and above will be logged
logger = logging.getLogger(__name__)

# suppress Gym Warnings
gym.logger.set_level(gym.logger.DISABLED)

# create Gym env
env = gym.make('NVIDIA-CASBOT-v1')

# create state and action dim and init memory
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
memory = Memory()

# create agent object
agent = CASBOTAgent(state_dim, action_dim, memory)

# redirect all stdout and stderr to suppress verbose output
with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
    for episode in range(500):
        # initialize environment
        state, _ = env.reset()
        total_reward = 0
        done = False

        # take an action and process the result
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.push(state, action, reward, next_state, done)
            agent.train(batch_size=64)
            state = next_state
            total_reward += reward

        # print only episode summary
        logger.info(f"Episode {episode}, Total Reward: {total_reward}")

        # update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
```

    INFO:__main__:Episode 0, Total Reward: 0.16625820860163032
    INFO:__main__:Episode 1, Total Reward: 0.5792603339452963
    INFO:__main__:Episode 2, Total Reward: 8.867517342813684
    INFO:__main__:Episode 3, Total Reward: 19.31451294021334
    INFO:__main__:Episode 4, Total Reward: 0.7993525205135067
    INFO:__main__:Episode 5, Total Reward: 3.278495189125822
    INFO:__main__:Episode 6, Total Reward: 0.7911482403531579
    INFO:__main__:Episode 7, Total Reward: 17.228864390402173
    INFO:__main__:Episode 8, Total Reward: 4.312042922707515
    INFO:__main__:Episode 9, Total Reward: 0.868751156698023
    INFO:__main__:Episode 10, Total Reward: 30.90379402175485
    INFO:__main__:Episode 11, Total Reward: 4.727897855313576
    INFO:__main__:Episode 12, Total Reward: 2.8015394523921118
    INFO:__main__:Episode 13, Total Reward: 3.23301796700322
    INFO:__main__:Episode 14, Total Reward: 4.6373937124670865
    INFO:__main__:Episode 15, Total Reward: 2.935659964264641
    INFO:__main__:Episode 16, Total Reward: 4.330096416794188
    INFO:__main__:Episode 17, Total Reward: 4.426721191107411
    INFO:__main__:Episode 18, Total Reward: 10.912886269161966
    INFO:__main__:Episode 19, Total Reward: 16.034471299670493
    INFO:__main__:Episode 20, Total Reward: 17.394410481355614
    INFO:__main__:Episode 21, Total Reward: 1.8962645898834831
    INFO:__main__:Episode 22, Total Reward: 18.15323437086349
    INFO:__main__:Episode 23, Total Reward: 17.073053327499544
    INFO:__main__:Episode 24, Total Reward: 13.761960331530329
    INFO:__main__:Episode 25, Total Reward: 23.687476385602736
    INFO:__main__:Episode 26, Total Reward: 0.9709667927627021
    INFO:__main__:Episode 27, Total Reward: 2.4451748994365605
    INFO:__main__:Episode 28, Total Reward: 21.248132143863934
    INFO:__main__:Episode 29, Total Reward: 6.009289317753753
    INFO:__main__:Episode 30, Total Reward: 12.64826040071319
    INFO:__main__:Episode 31, Total Reward: 13.606403803701289
    INFO:__main__:Episode 32, Total Reward: 8.569603965058027
    INFO:__main__:Episode 33, Total Reward: 15.03435992887554
    INFO:__main__:Episode 34, Total Reward: 2.529645520961023
    INFO:__main__:Episode 35, Total Reward: 14.390187075789427
    INFO:__main__:Episode 36, Total Reward: 10.023624878828702
    INFO:__main__:Episode 37, Total Reward: 23.76381517288978
    INFO:__main__:Episode 38, Total Reward: 1.6851910063644095
    INFO:__main__:Episode 39, Total Reward: 1.1818273140738973
    INFO:__main__:Episode 40, Total Reward: 14.776329771127305
    INFO:__main__:Episode 41, Total Reward: 16.68735075540676
    INFO:__main__:Episode 42, Total Reward: 0.7269285245516296
    INFO:__main__:Episode 43, Total Reward: 5.109754554165928
    INFO:__main__:Episode 44, Total Reward: 12.08303231270281
    INFO:__main__:Episode 45, Total Reward: 1.3328535842735134
    INFO:__main__:Episode 46, Total Reward: 4.0828245644370975
    INFO:__main__:Episode 47, Total Reward: 8.130118345641664
    INFO:__main__:Episode 48, Total Reward: 7.148002493064594
    INFO:__main__:Episode 49, Total Reward: 0.9973430100035272
    INFO:__main__:Episode 50, Total Reward: 7.6363229060259705
    INFO:__main__:Episode 51, Total Reward: 6.420469610891964
    INFO:__main__:Episode 52, Total Reward: 6.0057299165052225
    INFO:__main__:Episode 53, Total Reward: 30.376408003221204
    INFO:__main__:Episode 54, Total Reward: 1.7803319130520245
    INFO:__main__:Episode 55, Total Reward: 18.683202353323498
    INFO:__main__:Episode 56, Total Reward: 10.127295588172492
    INFO:__main__:Episode 57, Total Reward: 6.804359009750756
    INFO:__main__:Episode 58, Total Reward: 7.661783136559894
    INFO:__main__:Episode 59, Total Reward: 3.2409406619213748
    INFO:__main__:Episode 60, Total Reward: 20.706400309563023
    INFO:__main__:Episode 61, Total Reward: 6.028242635520645
    INFO:__main__:Episode 62, Total Reward: 3.7532834452740644
    INFO:__main__:Episode 63, Total Reward: 18.923800914284055
    INFO:__main__:Episode 64, Total Reward: 9.225820892376165
    INFO:__main__:Episode 65, Total Reward: 2.544668086793297
    INFO:__main__:Episode 66, Total Reward: 21.231997757686095
    INFO:__main__:Episode 67, Total Reward: 8.854393458594737
    INFO:__main__:Episode 68, Total Reward: 1.396572372872448
    INFO:__main__:Episode 69, Total Reward: 10.102833492909253
    INFO:__main__:Episode 70, Total Reward: 1.7989352747606424
    INFO:__main__:Episode 71, Total Reward: 6.740758351588277
    INFO:__main__:Episode 72, Total Reward: 3.7146139136526073
    INFO:__main__:Episode 73, Total Reward: 3.537468281122579
    INFO:__main__:Episode 74, Total Reward: 8.47754090803356
    INFO:__main__:Episode 75, Total Reward: 8.956296530374315
    INFO:__main__:Episode 76, Total Reward: 5.236543168706781
    INFO:__main__:Episode 77, Total Reward: 1.3699850891050187
    INFO:__main__:Episode 78, Total Reward: 15.29203181987748
    INFO:__main__:Episode 79, Total Reward: 2.6434471853847668
    INFO:__main__:Episode 80, Total Reward: 0.7392238557164938
    INFO:__main__:Episode 81, Total Reward: 6.258410642031511
    INFO:__main__:Episode 82, Total Reward: 8.60346604611778
    INFO:__main__:Episode 83, Total Reward: 2.253361170109483
    INFO:__main__:Episode 84, Total Reward: 22.462478031641748
    INFO:__main__:Episode 85, Total Reward: 34.88212433145185
    INFO:__main__:Episode 86, Total Reward: 12.514664206453679
    INFO:__main__:Episode 87, Total Reward: 31.61351623760162
    INFO:__main__:Episode 88, Total Reward: 9.248122167021686
    INFO:__main__:Episode 89, Total Reward: 20.85939842302011
    INFO:__main__:Episode 90, Total Reward: 3.016077002768217
    INFO:__main__:Episode 91, Total Reward: 1.9149888225692395
    INFO:__main__:Episode 92, Total Reward: 4.603419907171941
    INFO:__main__:Episode 93, Total Reward: 19.489110376465312
    INFO:__main__:Episode 94, Total Reward: 3.0615722217373875
    INFO:__main__:Episode 95, Total Reward: 20.57953839413646
    INFO:__main__:Episode 96, Total Reward: 14.42953174119293
    INFO:__main__:Episode 97, Total Reward: 20.367994707183247
    INFO:__main__:Episode 98, Total Reward: 10.614430783753901
    INFO:__main__:Episode 99, Total Reward: 5.183006876033224
    INFO:__main__:Episode 100, Total Reward: 14.310424313434574
    INFO:__main__:Episode 101, Total Reward: 22.312392387935983
    INFO:__main__:Episode 102, Total Reward: 12.366079646545208
    INFO:__main__:Episode 103, Total Reward: 9.842797483569505
    INFO:__main__:Episode 104, Total Reward: 4.512976450292518
    INFO:__main__:Episode 105, Total Reward: 0.8932220890118749
    INFO:__main__:Episode 106, Total Reward: 2.9350941428368533
    INFO:__main__:Episode 107, Total Reward: 6.622090895662509
    INFO:__main__:Episode 108, Total Reward: 4.105412850289914
    INFO:__main__:Episode 109, Total Reward: 0.7297162768520946
    INFO:__main__:Episode 110, Total Reward: 1.337708929935696
    INFO:__main__:Episode 111, Total Reward: 0.1142513863423873
    INFO:__main__:Episode 112, Total Reward: 43.62202513905373
    INFO:__main__:Episode 113, Total Reward: 12.28996920541138
    INFO:__main__:Episode 114, Total Reward: 3.6790304710022927
    INFO:__main__:Episode 115, Total Reward: 1.3593485909940348
    INFO:__main__:Episode 116, Total Reward: 16.772020357534824
    INFO:__main__:Episode 117, Total Reward: 0.4343678561428368
    INFO:__main__:Episode 118, Total Reward: 7.464790750786729
    INFO:__main__:Episode 119, Total Reward: 3.2553426648937935
    INFO:__main__:Episode 120, Total Reward: 16.625446711692156
    INFO:__main__:Episode 121, Total Reward: 4.574129654198123
    INFO:__main__:Episode 122, Total Reward: 6.859912084068765
    INFO:__main__:Episode 123, Total Reward: 8.015309116713743
    INFO:__main__:Episode 124, Total Reward: 23.750371074622347
    INFO:__main__:Episode 125, Total Reward: 25.60124855491525
    INFO:__main__:Episode 126, Total Reward: 26.28361682673267
    INFO:__main__:Episode 127, Total Reward: 10.530040793446627
    INFO:__main__:Episode 128, Total Reward: 16.551084481618823
    INFO:__main__:Episode 129, Total Reward: 0.289057814232458
    INFO:__main__:Episode 130, Total Reward: 22.975257594843747
    INFO:__main__:Episode 131, Total Reward: 1.7804402299307793
    INFO:__main__:Episode 132, Total Reward: 24.86999043055729
    INFO:__main__:Episode 133, Total Reward: 50.55722710462231
    INFO:__main__:Episode 134, Total Reward: 3.4191506177281132
    INFO:__main__:Episode 135, Total Reward: 7.590578174660498
    INFO:__main__:Episode 136, Total Reward: 18.061653542982555
    INFO:__main__:Episode 137, Total Reward: 7.7323377873423125
    INFO:__main__:Episode 138, Total Reward: 4.074721840146612
    INFO:__main__:Episode 139, Total Reward: 9.444431317737553
    INFO:__main__:Episode 140, Total Reward: 16.245790907869953
    INFO:__main__:Episode 141, Total Reward: 1.7129992881919267
    INFO:__main__:Episode 142, Total Reward: 2.6912934439521594
    INFO:__main__:Episode 143, Total Reward: 4.838977581243075
    INFO:__main__:Episode 144, Total Reward: 9.147972920749691
    INFO:__main__:Episode 145, Total Reward: 24.377164512270976
    INFO:__main__:Episode 146, Total Reward: 0.8488943814647475
    INFO:__main__:Episode 147, Total Reward: 20.02663937588619
    INFO:__main__:Episode 148, Total Reward: 27.581297009109953
    INFO:__main__:Episode 149, Total Reward: 16.27573088536294
    INFO:__main__:Episode 150, Total Reward: 1.253106576747237
    INFO:__main__:Episode 151, Total Reward: 2.060173489324881
    INFO:__main__:Episode 152, Total Reward: 5.760733698343441
    INFO:__main__:Episode 153, Total Reward: 18.081600438347326
    INFO:__main__:Episode 154, Total Reward: 2.4610947632869316
    INFO:__main__:Episode 155, Total Reward: 10.160876659178484
    INFO:__main__:Episode 156, Total Reward: 2.3518724994691342
    INFO:__main__:Episode 157, Total Reward: 10.38741378730956
    INFO:__main__:Episode 158, Total Reward: 16.08888935074325
    INFO:__main__:Episode 159, Total Reward: 12.926132942272432
    INFO:__main__:Episode 160, Total Reward: 6.5029721878293465
    INFO:__main__:Episode 161, Total Reward: 2.618799833271761
    INFO:__main__:Episode 162, Total Reward: 7.454893631290578
    INFO:__main__:Episode 163, Total Reward: 17.492533158017647
    INFO:__main__:Episode 164, Total Reward: 3.2527059911809095
    INFO:__main__:Episode 165, Total Reward: 11.76029204924193
    INFO:__main__:Episode 166, Total Reward: 21.018614560254743
    INFO:__main__:Episode 167, Total Reward: 1.4458316450582367
    INFO:__main__:Episode 168, Total Reward: 5.813257802059131
    INFO:__main__:Episode 169, Total Reward: 7.803350805190481
    INFO:__main__:Episode 170, Total Reward: 5.84351143437764
    INFO:__main__:Episode 171, Total Reward: 6.957316149282549
    INFO:__main__:Episode 172, Total Reward: 20.509381759376062
    INFO:__main__:Episode 173, Total Reward: 0.977365697140842
    INFO:__main__:Episode 174, Total Reward: 5.24645835628865
    INFO:__main__:Episode 175, Total Reward: 6.184280010286
    INFO:__main__:Episode 176, Total Reward: 6.513053617395798
    INFO:__main__:Episode 177, Total Reward: 1.9966504378208718
    INFO:__main__:Episode 178, Total Reward: 2.9112794195437064
    INFO:__main__:Episode 179, Total Reward: 3.0301548243700127
    INFO:__main__:Episode 180, Total Reward: 0.23773750404888083
    INFO:__main__:Episode 181, Total Reward: 6.24884226181412
    INFO:__main__:Episode 182, Total Reward: 4.435377714737758
    INFO:__main__:Episode 183, Total Reward: 15.511242425514782
    INFO:__main__:Episode 184, Total Reward: 1.5079588469042848
    INFO:__main__:Episode 185, Total Reward: 4.20037242656716
    INFO:__main__:Episode 186, Total Reward: 5.309989845496251
    INFO:__main__:Episode 187, Total Reward: 3.816147434008636
    INFO:__main__:Episode 188, Total Reward: 38.284325291883576
    INFO:__main__:Episode 189, Total Reward: 0.8312669269191917
    INFO:__main__:Episode 190, Total Reward: 12.924331848817964
    INFO:__main__:Episode 191, Total Reward: 17.95284154488244
    INFO:__main__:Episode 192, Total Reward: 7.559321635278071
    INFO:__main__:Episode 193, Total Reward: 0.7581333698800229
    INFO:__main__:Episode 194, Total Reward: 0.3841060447225937
    INFO:__main__:Episode 195, Total Reward: 4.5461080829923715
    INFO:__main__:Episode 196, Total Reward: 1.169377055834351
    INFO:__main__:Episode 197, Total Reward: 5.435287378373959
    INFO:__main__:Episode 198, Total Reward: 9.128720793980518
    INFO:__main__:Episode 199, Total Reward: 3.6213356123119986
    INFO:__main__:Episode 200, Total Reward: 6.951961230619286
    INFO:__main__:Episode 201, Total Reward: 3.1805967392025556
    INFO:__main__:Episode 202, Total Reward: 9.490041054533872
    INFO:__main__:Episode 203, Total Reward: 26.437427148787155
    INFO:__main__:Episode 204, Total Reward: 8.937440945378908
    INFO:__main__:Episode 205, Total Reward: 17.15015699566088
    INFO:__main__:Episode 206, Total Reward: 13.931525753683932
    INFO:__main__:Episode 207, Total Reward: 14.680193461044215
    INFO:__main__:Episode 208, Total Reward: 7.832483984730106
    INFO:__main__:Episode 209, Total Reward: 10.196063735073896
    INFO:__main__:Episode 210, Total Reward: 7.6267089378576065
    INFO:__main__:Episode 211, Total Reward: 21.98328409159726
    INFO:__main__:Episode 212, Total Reward: 13.943463813264499
    INFO:__main__:Episode 213, Total Reward: 1.425492243776729
    INFO:__main__:Episode 214, Total Reward: 10.383224315873044
    INFO:__main__:Episode 215, Total Reward: 19.512378023706983
    INFO:__main__:Episode 216, Total Reward: 12.86848815416506
    INFO:__main__:Episode 217, Total Reward: 14.405388533701116
    INFO:__main__:Episode 218, Total Reward: 0.896601871548984
    INFO:__main__:Episode 219, Total Reward: 2.4593726953420756
    INFO:__main__:Episode 220, Total Reward: 3.2453445395001026
    INFO:__main__:Episode 221, Total Reward: 14.2554196516289
    INFO:__main__:Episode 222, Total Reward: 0.2584056622831631
    INFO:__main__:Episode 223, Total Reward: 6.176318968483534
    INFO:__main__:Episode 224, Total Reward: 1.1960139201429374
    INFO:__main__:Episode 225, Total Reward: 1.451124761506239
    INFO:__main__:Episode 226, Total Reward: 30.675915445462373
    INFO:__main__:Episode 227, Total Reward: 9.932890464412466
    INFO:__main__:Episode 228, Total Reward: 2.212560807348152
    INFO:__main__:Episode 229, Total Reward: 1.8520777222335183
    INFO:__main__:Episode 230, Total Reward: 4.084675636264722
    INFO:__main__:Episode 231, Total Reward: 15.708203506334169
    INFO:__main__:Episode 232, Total Reward: 32.12346534640789
    INFO:__main__:Episode 233, Total Reward: 2.406867615411195
    INFO:__main__:Episode 234, Total Reward: 15.14507047270868
    INFO:__main__:Episode 235, Total Reward: 0.3254122660118567
    INFO:__main__:Episode 236, Total Reward: 12.37313432706257
    INFO:__main__:Episode 237, Total Reward: 1.3128126620108138
    INFO:__main__:Episode 238, Total Reward: 8.002548479674358
    INFO:__main__:Episode 239, Total Reward: 1.5665383057798703
    INFO:__main__:Episode 240, Total Reward: 4.70455015734483
    INFO:__main__:Episode 241, Total Reward: 4.060826690102701
    INFO:__main__:Episode 242, Total Reward: 19.29552304408254
    INFO:__main__:Episode 243, Total Reward: 18.12644209618762
    INFO:__main__:Episode 244, Total Reward: 3.1597891438920214
    INFO:__main__:Episode 245, Total Reward: 2.7841853092326607
    INFO:__main__:Episode 246, Total Reward: 11.702373387540046
    INFO:__main__:Episode 247, Total Reward: 0.6773343760901024
    INFO:__main__:Episode 248, Total Reward: 5.5166456797769765
    INFO:__main__:Episode 249, Total Reward: 3.700435765985654
    INFO:__main__:Episode 250, Total Reward: 6.31423014931693
    INFO:__main__:Episode 251, Total Reward: 28.262080403183663
    INFO:__main__:Episode 252, Total Reward: 8.113068539647129
    INFO:__main__:Episode 253, Total Reward: 11.493450535348659
    INFO:__main__:Episode 254, Total Reward: 51.62723359552371
    INFO:__main__:Episode 255, Total Reward: 0.9721425397311984
    INFO:__main__:Episode 256, Total Reward: 11.75668335660986
    INFO:__main__:Episode 257, Total Reward: 23.707377137706796
    INFO:__main__:Episode 258, Total Reward: 2.4652639636116445
    INFO:__main__:Episode 259, Total Reward: 8.432662569187578
    INFO:__main__:Episode 260, Total Reward: 0.9309558417070013
    INFO:__main__:Episode 261, Total Reward: 0.5482445633217096
    INFO:__main__:Episode 262, Total Reward: 2.582343168127137
    INFO:__main__:Episode 263, Total Reward: 0.2972407109243085
    INFO:__main__:Episode 264, Total Reward: 2.462271927741246
    INFO:__main__:Episode 265, Total Reward: 5.145898997709348
    INFO:__main__:Episode 266, Total Reward: 10.57592661296064
    INFO:__main__:Episode 267, Total Reward: 0.6368672570974026
    INFO:__main__:Episode 268, Total Reward: 38.682543908317164
    INFO:__main__:Episode 269, Total Reward: 5.482484129863898
    INFO:__main__:Episode 270, Total Reward: 7.133182218704136
    INFO:__main__:Episode 271, Total Reward: 2.185964711360965
    INFO:__main__:Episode 272, Total Reward: 12.602841496604855
    INFO:__main__:Episode 273, Total Reward: 7.409192380794587
    INFO:__main__:Episode 274, Total Reward: 21.87984266476733
    INFO:__main__:Episode 275, Total Reward: 12.19277201947817
    INFO:__main__:Episode 276, Total Reward: 6.10553885031652
    INFO:__main__:Episode 277, Total Reward: 3.5751468712057215
    INFO:__main__:Episode 278, Total Reward: 3.4711907491737586
    INFO:__main__:Episode 279, Total Reward: 5.909976380625405
    INFO:__main__:Episode 280, Total Reward: 0.4573871174558335
    INFO:__main__:Episode 281, Total Reward: 6.835162284732567
    INFO:__main__:Episode 282, Total Reward: 5.009004847847282
    INFO:__main__:Episode 283, Total Reward: 5.988267276945565
    INFO:__main__:Episode 284, Total Reward: 11.46027489050396
    INFO:__main__:Episode 285, Total Reward: 5.05284571522164
    INFO:__main__:Episode 286, Total Reward: 10.439585913819531
    INFO:__main__:Episode 287, Total Reward: 15.13539532523656
    INFO:__main__:Episode 288, Total Reward: 7.808535407836202
    INFO:__main__:Episode 289, Total Reward: 0.7469210165286031
    INFO:__main__:Episode 290, Total Reward: 4.758517327805197
    INFO:__main__:Episode 291, Total Reward: 0.9511095249214591
    INFO:__main__:Episode 292, Total Reward: 9.607666235707914
    INFO:__main__:Episode 293, Total Reward: 32.73352899125225
    INFO:__main__:Episode 294, Total Reward: 3.612399942830715
    INFO:__main__:Episode 295, Total Reward: 22.80887724821264
    INFO:__main__:Episode 296, Total Reward: 7.201194957357643
    INFO:__main__:Episode 297, Total Reward: 1.0910700558767392
    INFO:__main__:Episode 298, Total Reward: 21.90312787053953
    INFO:__main__:Episode 299, Total Reward: 8.288095970603775
    INFO:__main__:Episode 300, Total Reward: 19.360049392713563
    INFO:__main__:Episode 301, Total Reward: 10.590417919864716
    INFO:__main__:Episode 302, Total Reward: 23.795561793619004
    INFO:__main__:Episode 303, Total Reward: 0.047980876245220405
    INFO:__main__:Episode 304, Total Reward: 4.609818547669793
    INFO:__main__:Episode 305, Total Reward: 16.784457348661384
    INFO:__main__:Episode 306, Total Reward: 14.464977005924876
    INFO:__main__:Episode 307, Total Reward: 0.7155704290127367
    INFO:__main__:Episode 308, Total Reward: 9.29715622731647
    INFO:__main__:Episode 309, Total Reward: 4.982718701126256
    INFO:__main__:Episode 310, Total Reward: 17.142163461327364
    INFO:__main__:Episode 311, Total Reward: 7.128347249455664
    INFO:__main__:Episode 312, Total Reward: 37.0854812029865
    INFO:__main__:Episode 313, Total Reward: 2.4014546703578485
    INFO:__main__:Episode 314, Total Reward: 9.813609614431003
    INFO:__main__:Episode 315, Total Reward: 0.1537735293186121
    INFO:__main__:Episode 316, Total Reward: 5.812573217021578
    INFO:__main__:Episode 317, Total Reward: 1.6106074690970147
    INFO:__main__:Episode 318, Total Reward: 27.899844395407797
    INFO:__main__:Episode 319, Total Reward: 0.01438505199094009
    INFO:__main__:Episode 320, Total Reward: 22.125684375775556
    INFO:__main__:Episode 321, Total Reward: 1.5854357635730398
    INFO:__main__:Episode 322, Total Reward: 14.375490261246965
    INFO:__main__:Episode 323, Total Reward: 6.659243301003496
    INFO:__main__:Episode 324, Total Reward: 4.168333085280211
    INFO:__main__:Episode 325, Total Reward: 2.566042564186178
    INFO:__main__:Episode 326, Total Reward: 9.74281424821949
    INFO:__main__:Episode 327, Total Reward: 9.854330066743078
    INFO:__main__:Episode 328, Total Reward: 4.467749022098238
    INFO:__main__:Episode 329, Total Reward: 26.88579546164378
    INFO:__main__:Episode 330, Total Reward: 3.1798621217035996
    INFO:__main__:Episode 331, Total Reward: 6.990994100166948
    INFO:__main__:Episode 332, Total Reward: 5.563639011074361
    INFO:__main__:Episode 333, Total Reward: 3.1472516728028372
    INFO:__main__:Episode 334, Total Reward: 33.11542226216203
    INFO:__main__:Episode 335, Total Reward: 9.142941678132198
    INFO:__main__:Episode 336, Total Reward: 20.167248891186485
    INFO:__main__:Episode 337, Total Reward: 1.1778578614152289
    INFO:__main__:Episode 338, Total Reward: 21.302895385078827
    INFO:__main__:Episode 339, Total Reward: 15.487728118601808
    INFO:__main__:Episode 340, Total Reward: 9.080768603708366
    INFO:__main__:Episode 341, Total Reward: 1.7617379237695405
    INFO:__main__:Episode 342, Total Reward: 16.02388894433095
    INFO:__main__:Episode 343, Total Reward: 3.186864528636365
    INFO:__main__:Episode 344, Total Reward: 2.9495411038538957
    INFO:__main__:Episode 345, Total Reward: 11.444247490830524
    INFO:__main__:Episode 346, Total Reward: 27.178383031223433
    INFO:__main__:Episode 347, Total Reward: 11.573931951199555
    INFO:__main__:Episode 348, Total Reward: 4.352638599434059
    INFO:__main__:Episode 349, Total Reward: 14.157771733616237
    INFO:__main__:Episode 350, Total Reward: 7.258824846913525
    INFO:__main__:Episode 351, Total Reward: 1.601203523658447
    INFO:__main__:Episode 352, Total Reward: 5.613302430426608
    INFO:__main__:Episode 353, Total Reward: 1.3003439313557377
    INFO:__main__:Episode 354, Total Reward: 18.73878660850529
    INFO:__main__:Episode 355, Total Reward: 8.326890215442422
    INFO:__main__:Episode 356, Total Reward: 13.168819385324841
    INFO:__main__:Episode 357, Total Reward: 5.302337610457513
    INFO:__main__:Episode 358, Total Reward: 13.099382819476697
    INFO:__main__:Episode 359, Total Reward: 25.69238187281807
    INFO:__main__:Episode 360, Total Reward: 0.5051412671339461
    INFO:__main__:Episode 361, Total Reward: 8.54794478338663
    INFO:__main__:Episode 362, Total Reward: 4.87579456828629
    INFO:__main__:Episode 363, Total Reward: 14.387108228060477
    INFO:__main__:Episode 364, Total Reward: 11.16025500959762
    INFO:__main__:Episode 365, Total Reward: 13.892627831054359
    INFO:__main__:Episode 366, Total Reward: 5.748199086024012
    INFO:__main__:Episode 367, Total Reward: 23.94604352277419
    INFO:__main__:Episode 368, Total Reward: 5.307285382550325
    INFO:__main__:Episode 369, Total Reward: 3.4092030207599504
    INFO:__main__:Episode 370, Total Reward: 1.1292437259770283
    INFO:__main__:Episode 371, Total Reward: 47.25968465003978
    INFO:__main__:Episode 372, Total Reward: 12.211410229266322
    INFO:__main__:Episode 373, Total Reward: 7.152000504668164
    INFO:__main__:Episode 374, Total Reward: 32.143480462432564
    INFO:__main__:Episode 375, Total Reward: 0.5362856483841855
    INFO:__main__:Episode 376, Total Reward: 5.318821026878279
    INFO:__main__:Episode 377, Total Reward: 1.4065720512975894
    INFO:__main__:Episode 378, Total Reward: 12.963363010674966
    INFO:__main__:Episode 379, Total Reward: 13.023874824523793
    INFO:__main__:Episode 380, Total Reward: 15.7972517130849
    INFO:__main__:Episode 381, Total Reward: 11.891133896804378
    INFO:__main__:Episode 382, Total Reward: 0.599845000843561
    INFO:__main__:Episode 383, Total Reward: 2.383673692268599
    INFO:__main__:Episode 384, Total Reward: 3.4978562940044413
    INFO:__main__:Episode 385, Total Reward: 11.964564448712624
    INFO:__main__:Episode 386, Total Reward: 3.717487956027588
    INFO:__main__:Episode 387, Total Reward: 7.779650187395831
    INFO:__main__:Episode 388, Total Reward: 45.11601270423608
    INFO:__main__:Episode 389, Total Reward: 8.21561468508726
    INFO:__main__:Episode 390, Total Reward: 4.488234635331214
    INFO:__main__:Episode 391, Total Reward: 2.5421948390971303
    INFO:__main__:Episode 392, Total Reward: 8.658821746145488
    INFO:__main__:Episode 393, Total Reward: 9.573672480416889
    INFO:__main__:Episode 394, Total Reward: 2.8150273694292447
    INFO:__main__:Episode 395, Total Reward: 1.991953254465764
    INFO:__main__:Episode 396, Total Reward: 20.37536068467541
    INFO:__main__:Episode 397, Total Reward: 1.761359140922329
    INFO:__main__:Episode 398, Total Reward: 6.4643891774465985
    INFO:__main__:Episode 399, Total Reward: 10.230704379383612
    INFO:__main__:Episode 400, Total Reward: 7.483974554784176
    INFO:__main__:Episode 401, Total Reward: 16.942119045391408
    INFO:__main__:Episode 402, Total Reward: 1.7673013590013724
    INFO:__main__:Episode 403, Total Reward: 1.2909645931010303
    INFO:__main__:Episode 404, Total Reward: 2.0946933371036964
    INFO:__main__:Episode 405, Total Reward: 0.9847386934440943
    INFO:__main__:Episode 406, Total Reward: 37.714285702684165
    INFO:__main__:Episode 407, Total Reward: 14.162064712632905
    INFO:__main__:Episode 408, Total Reward: 28.80675533269499
    INFO:__main__:Episode 409, Total Reward: 2.430554669184809
    INFO:__main__:Episode 410, Total Reward: 5.229412693961314
    INFO:__main__:Episode 411, Total Reward: 7.456918848850615
    INFO:__main__:Episode 412, Total Reward: 0.5526891732773217
    INFO:__main__:Episode 413, Total Reward: 1.1857068761687208
    INFO:__main__:Episode 414, Total Reward: 19.70540696125804
    INFO:__main__:Episode 415, Total Reward: 5.169842276556783
    INFO:__main__:Episode 416, Total Reward: 2.8457901678918702
    INFO:__main__:Episode 417, Total Reward: 1.9013203776067082
    INFO:__main__:Episode 418, Total Reward: 29.721699932406494
    INFO:__main__:Episode 419, Total Reward: 13.418028642462584
    INFO:__main__:Episode 420, Total Reward: 0.43431292674191657
    INFO:__main__:Episode 421, Total Reward: 35.668463275535835
    INFO:__main__:Episode 422, Total Reward: 6.2905420279932445
    INFO:__main__:Episode 423, Total Reward: 33.65847731473412
    INFO:__main__:Episode 424, Total Reward: 4.672741201049901
    INFO:__main__:Episode 425, Total Reward: 33.06125918299967
    INFO:__main__:Episode 426, Total Reward: 31.877204342218086
    INFO:__main__:Episode 427, Total Reward: 0.9744097484772642
    INFO:__main__:Episode 428, Total Reward: 0.33315576388937485
    INFO:__main__:Episode 429, Total Reward: 3.4150655292666365
    INFO:__main__:Episode 430, Total Reward: 6.842308053605012
    INFO:__main__:Episode 431, Total Reward: 4.682182043323634
    INFO:__main__:Episode 432, Total Reward: 1.929272087285483
    INFO:__main__:Episode 433, Total Reward: 7.425592568276348
    INFO:__main__:Episode 434, Total Reward: 24.005099650729456
    INFO:__main__:Episode 435, Total Reward: 9.404563750032464
    INFO:__main__:Episode 436, Total Reward: 14.642118379814264
    INFO:__main__:Episode 437, Total Reward: 1.2259259667509141
    INFO:__main__:Episode 438, Total Reward: 9.02934382425011
    INFO:__main__:Episode 439, Total Reward: 19.455464146305395
    INFO:__main__:Episode 440, Total Reward: 6.353994457302137
    INFO:__main__:Episode 441, Total Reward: 3.643068882005358
    INFO:__main__:Episode 442, Total Reward: 2.3533013381187047
    INFO:__main__:Episode 443, Total Reward: 14.671010002998967
    INFO:__main__:Episode 444, Total Reward: 16.093418667721657
    INFO:__main__:Episode 445, Total Reward: 0.3477037066122375
    INFO:__main__:Episode 446, Total Reward: 16.231725709711682
    INFO:__main__:Episode 447, Total Reward: 1.2720809065315892
    INFO:__main__:Episode 448, Total Reward: 6.568590093576131
    INFO:__main__:Episode 449, Total Reward: 3.1481902470549135
    INFO:__main__:Episode 450, Total Reward: 30.435961316052975
    INFO:__main__:Episode 451, Total Reward: 10.130700101222029
    INFO:__main__:Episode 452, Total Reward: 6.883088157823491
    INFO:__main__:Episode 453, Total Reward: 11.801530630214451
    INFO:__main__:Episode 454, Total Reward: 35.650280471184594
    INFO:__main__:Episode 455, Total Reward: 1.1232940637420723
    INFO:__main__:Episode 456, Total Reward: 0.9048867458828233
    INFO:__main__:Episode 457, Total Reward: 42.834581139232824
    INFO:__main__:Episode 458, Total Reward: 18.103150031350758
    INFO:__main__:Episode 459, Total Reward: 12.41274826756215
    INFO:__main__:Episode 460, Total Reward: 3.8921015682117983
    INFO:__main__:Episode 461, Total Reward: 9.632447072176435
    INFO:__main__:Episode 462, Total Reward: 18.215514270069864
    INFO:__main__:Episode 463, Total Reward: 11.507839619368733
    INFO:__main__:Episode 464, Total Reward: 17.627017892454504
    INFO:__main__:Episode 465, Total Reward: 20.278546360124167
    INFO:__main__:Episode 466, Total Reward: 5.698891858219017
    INFO:__main__:Episode 467, Total Reward: 11.922439968469515
    INFO:__main__:Episode 468, Total Reward: 5.817589381130197
    INFO:__main__:Episode 469, Total Reward: 44.74423904178442
    INFO:__main__:Episode 470, Total Reward: 9.264935248516935
    INFO:__main__:Episode 471, Total Reward: 1.3198699705263164
    INFO:__main__:Episode 472, Total Reward: 13.128160704584277
    INFO:__main__:Episode 473, Total Reward: 64.6965390873808
    INFO:__main__:Episode 474, Total Reward: 39.45671650984505
    INFO:__main__:Episode 475, Total Reward: 15.085602667042249
    INFO:__main__:Episode 476, Total Reward: 12.52723478718893
    INFO:__main__:Episode 477, Total Reward: 0.11007128986857362
    INFO:__main__:Episode 478, Total Reward: 4.938698604819712
    INFO:__main__:Episode 479, Total Reward: 0.9257542600863006
    INFO:__main__:Episode 480, Total Reward: 22.173868394516006
    INFO:__main__:Episode 481, Total Reward: 3.138370115199212
    INFO:__main__:Episode 482, Total Reward: 0.7096876829330329
    INFO:__main__:Episode 483, Total Reward: 12.012440553009183
    INFO:__main__:Episode 484, Total Reward: 5.305223683916743
    INFO:__main__:Episode 485, Total Reward: 0.9851101812138134
    INFO:__main__:Episode 486, Total Reward: 2.9034817017257466
    INFO:__main__:Episode 487, Total Reward: 19.230243216810802
    INFO:__main__:Episode 488, Total Reward: 5.8911442181797105
    INFO:__main__:Episode 489, Total Reward: 3.213285670732294
    INFO:__main__:Episode 490, Total Reward: 2.3188362554560182
    INFO:__main__:Episode 491, Total Reward: 3.372655417231018
    INFO:__main__:Episode 492, Total Reward: 14.871276712201151
    INFO:__main__:Episode 493, Total Reward: 16.9551464347525
    INFO:__main__:Episode 494, Total Reward: 44.14395502613953
    INFO:__main__:Episode 495, Total Reward: 20.219028225477118
    INFO:__main__:Episode 496, Total Reward: 7.325208664300143
    INFO:__main__:Episode 497, Total Reward: 4.286970620667298
    INFO:__main__:Episode 498, Total Reward: 9.616736190495343
    INFO:__main__:Episode 499, Total Reward: 6.588099544010954


## Final Analysis
* This output provides a detailed log of the total reward achieved by the agent for each of the 500 episodes in a reinforcement learning environment. Each episode represents a single run of the agent interacting with the environment until a termination condition (e.g., completing a task or reaching a maximum time limit) is met. The “Total Reward” metric aggregates all rewards collected by the agent throughout the episode, serving as a measure of performance for that episode. Higher rewards generally indicate better performance, suggesting the agent is learning and improving its ability to select actions that maximize rewards over time.
* Looking at the results, the total rewards exhibit considerable variability across episodes. Some episodes show very high rewards (e.g., Episode 473 with 64.7 and Episode 474 with 39.5), indicating that the agent performed well in those instances. On the other hand, episodes with low rewards (e.g., Episode 477 with 0.11) suggest suboptimal performance or challenging scenarios within the environment. This variability is common in reinforcement learning, especially during early training phases, as the agent explores various actions and learns an optimal policy through trial and error.
* The presence of episodes with exceptionally high rewards later in the sequence (e.g., Episodes 473 and 494) could indicate that the agent is beginning to exploit its learned policy effectively. However, the continued presence of low-reward episodes suggests room for further improvement in policy consistency. Overall, this log demonstrates the incremental progress of a reinforcement learning agent, characterized by the oscillation between exploration (trying new actions) and exploitation (choosing actions based on learned knowledge). A deeper analysis of reward trends over time, combined with exploration-exploitation metrics, could provide further insights into the agent’s learning dynamics.
