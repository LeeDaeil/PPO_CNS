import random
import numpy as np
import os
from Model_0_Basic.TOOL import Replay


class ReplayMemory(Replay.ReplayMemory):
    def __init__(self, capacity, nub_env):
        super(ReplayMemory, self).__init__(capacity=capacity)
        self.nub_env = nub_env
        self.gp_db = {
            f'{i}': {
                'KCNTOMS': [],

                'UUPPPL': [],   'UPRZ': [],
                'ZINST65': [],  'ZINST63': [],
                'BHV142': [],   'BFV122': [],   'ZINST66': [],

                'Reward': [],
            } for i in range(self.nub_env)
        }

    def push(self, state, action, reward, next_state, done):
        super(ReplayMemory, self).push(state, action, reward, next_state, done)

    def sample(self, batch_size):
        return super(ReplayMemory, self).sample(batch_size)

    def add_ep(self):
        return super(ReplayMemory, self).add_ep()

    def add_total_numsteps(self):
        return super(ReplayMemory, self).add_total_numsteps()

    def add_train_info(self, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha):
        super(ReplayMemory, self).add_train_info(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha)

    def get_len(self):
        return super(ReplayMemory, self).get_len()

    def get_ep(self):
        return super(ReplayMemory, self).get_ep()

    def get_total_numstps(self):
        return super(ReplayMemory, self).get_total_numstps()

    def get_train_info(self):
        return super(ReplayMemory, self).get_train_info()

    def get_finish_info(self):
        return super(ReplayMemory, self).get_finish_info()

    # ------------------------------------------------------------------------------------------------------------------
    # 추가된 기능
    def add_para(self, env_i, KCNTOMS, UUPPPL, UPRZ, ZINST65, ZINST63, BHV142, BFV122, ZINST66, Reward):
        self.gp_db[env_i]['KCNTOMS'].append(KCNTOMS)

        self.gp_db[env_i]['UUPPPL'].append(UUPPPL)
        self.gp_db[env_i]['UPRZ'].append(UPRZ)

        self.gp_db[env_i]['ZINST65'].append(ZINST65)
        self.gp_db[env_i]['ZINST63'].append(ZINST63)

        self.gp_db[env_i]['BHV142'].append(BHV142)
        self.gp_db[env_i]['BFV122'].append(BFV122)
        self.gp_db[env_i]['ZINST66'].append(ZINST66)

        self.gp_db[env_i]['Reward'].append(Reward)

    def get_para(self, env_i):
        KCNTOMS = self.gp_db[env_i]['KCNTOMS']

        UUPPPL = self.gp_db[env_i]['UUPPPL']
        UPRZ = self.gp_db[env_i]['UPRZ']

        ZINST65 = self.gp_db[env_i]['ZINST65']
        ZINST63 = self.gp_db[env_i]['ZINST63']

        BHV142 = self.gp_db[env_i]['BHV142']
        BFV122 = self.gp_db[env_i]['BFV122']
        ZINST66 = self.gp_db[env_i]['ZINST66']

        Reward = self.gp_db[env_i]['Reward']

        return KCNTOMS, UUPPPL, UPRZ, ZINST65, ZINST63, BHV142, BFV122, ZINST66, Reward

    def clear_para(self, env_i):
        self.gp_db[env_i] = {
                'KCNTOMS': [],

                'UUPPPL': [],   'UPRZ': [],
                'ZINST65': [],  'ZINST63': [],
                'BHV142': [],   'BFV122': [],   'ZINST66': [],

                'Reward': [],
            }
