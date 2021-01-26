import numpy as np
import torch

class SAC_Base:
    def __init__(self,
                 action_space,
                 observation_space,
                 replay_buffer,
                 lr, policy_type, automatic_entropy_tuning
                 ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.policy_type = policy_type
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = lr

        self.automatic_entropy_tuning = automatic_entropy_tuning
        # self.batch_size = 256
        self.batch_size = 64
        self.nub_steps = 1e6
        self.update_per_step = 1            # Every step net update
        self.update_target_per_step = 1     #

        # self.start_step = 1e4               # After start_step, action from agent
        self.start_step = 20               # After start_step, action from agent
        self.replay_buffer = replay_buffer  # Shared replay buffer

        self.cuda = False
        self.device = torch.device("cuda" if self.cuda else "cpu")
