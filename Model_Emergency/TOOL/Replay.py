import random
import numpy as np
import os
from Model_Basic.TOOL import Replay


class ReplayMemory(Replay.ReplayMemory):
    def __init__(self, capacity, nub_env):
        super(ReplayMemory, self).__init__(capacity=capacity)

        self.nub_env = nub_env
        self.gp_db = {
            f'{i}': {
                'KCNTOMS': [],
                'UAVLEG2': [],
                'ZINST65': [],

                'WFWLN123': [],
            } for i in range(self.nub_env)
        }


    def push(self, state, action, reward, next_state, done):
        super(ReplayMemory, self).push()

    def sample(self, batch_size):
        return super(ReplayMemory, self).sample()

    def add_ep(self):
        return super(ReplayMemory, self).add_ep()

    def add_total_numsteps(self):
        return super(ReplayMemory, self).add_total_numsteps()

    def add_train_info(self, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha):
        super(ReplayMemory, self).add_train_info()

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
    def add_para(self, env_i, ctime, avgtemp, pzrpres, allfeed):
        self.gp_db[env_i]['KCNTOMS'].append(ctime)
        self.gp_db[env_i]['UAVLEG2'].append(avgtemp)
        self.gp_db[env_i]['ZINST65'].append(pzrpres)
        self.gp_db[env_i]['WFWLN123'].append(allfeed)

    def get_para(self, env_i):
        KCNTOMS = self.gp_db[env_i]['KCNTOMS']
        UAVLEG2 = self.gp_db[env_i]['KCNTOMS']
        ZINST65 = self.gp_db[env_i]['KCNTOMS']
        WFWLN123 = self.gp_db[env_i]['KCNTOMS']
        return KCNTOMS, UAVLEG2, ZINST65, WFWLN123

    def clear_para(self, env_i):
        _ = [self.gp_db[env_i][_].clear for _ in self.gp_db[env_i].keys()]