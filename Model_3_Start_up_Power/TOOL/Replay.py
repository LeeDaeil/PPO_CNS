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

                'QPROREL': [],   'Ref_P': [],   'Ref_UpP': [],  'Ref_DoP': [],
                'rod_pos1': [],  'rod_pos2': [],  'rod_pos3': [],  'rod_pos4': [],

                'EBOAC': [], 'KBCDO16': [], 'WBOAC': [],

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
    def add_para(self, env_i, KCNTOMS, QPROREL, Ref_P, Ref_UpP, Ref_DoP,
                 rod_pos1, rod_pos2, rod_pos3, rod_pos4, EBOAC, KBCDO16, WBOAC,
                 Reward):
        self.gp_db[env_i]['KCNTOMS'].append(KCNTOMS)

        self.gp_db[env_i]['QPROREL'].append(QPROREL)
        self.gp_db[env_i]['Ref_P'].append(Ref_P)
        self.gp_db[env_i]['Ref_UpP'].append(Ref_UpP)
        self.gp_db[env_i]['Ref_DoP'].append(Ref_DoP)

        self.gp_db[env_i]['rod_pos1'].append(rod_pos1)
        self.gp_db[env_i]['rod_pos2'].append(rod_pos2)
        self.gp_db[env_i]['rod_pos3'].append(rod_pos3)
        self.gp_db[env_i]['rod_pos4'].append(rod_pos4)

        self.gp_db[env_i]['EBOAC'].append(EBOAC)
        self.gp_db[env_i]['KBCDO16'].append(KBCDO16)
        self.gp_db[env_i]['WBOAC'].append(WBOAC)

        self.gp_db[env_i]['Reward'].append(Reward)

    def get_para(self, env_i):
        KCNTOMS = self.gp_db[env_i]['KCNTOMS']

        QPROREL = self.gp_db[env_i]['QPROREL']
        Ref_P = self.gp_db[env_i]['Ref_P']
        Ref_UpP = self.gp_db[env_i]['Ref_UpP']
        Ref_DoP = self.gp_db[env_i]['Ref_DoP']

        rod_pos1 = self.gp_db[env_i]['rod_pos1']
        rod_pos2 = self.gp_db[env_i]['rod_pos2']
        rod_pos3 = self.gp_db[env_i]['rod_pos3']
        rod_pos4 = self.gp_db[env_i]['rod_pos4']

        EBOAC = self.gp_db[env_i]['EBOAC']
        KBCDO16 = self.gp_db[env_i]['KBCDO16']
        WBOAC = self.gp_db[env_i]['WBOAC']

        Reward = self.gp_db[env_i]['Reward']

        return KCNTOMS, QPROREL, Ref_P, Ref_UpP, Ref_DoP,\
               rod_pos1, rod_pos2, rod_pos3, rod_pos4, EBOAC, KBCDO16, WBOAC,\
               Reward

    def clear_para(self, env_i):
        self.gp_db[env_i] = {
                'KCNTOMS': [],

                'QPROREL': [],   'Ref_P': [],   'Ref_UpP': [],  'Ref_DoP': [],
                'rod_pos1': [],  'rod_pos2': [],  'rod_pos3': [],  'rod_pos4': [],

                'EBOAC': [], 'KBCDO16': [], 'WBOAC': [],

                'Reward': [],
            }
