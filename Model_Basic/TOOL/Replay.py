import random
import numpy as np
import os


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

        self.buffer_critic_1_loss = []
        self.buffer_critic_2_loss = []
        self.buffer_policy_loss = []
        self.buffer_ent_loss = []
        self.buffer_alpha = []

        self.position = 0

        self.total_ep = 0

        self.total_numsteps = 0

        self.end_numsteps = 100

        print(f'[{"Replay Memory Info":20}][Capacity|{self.capacity}][End_Stes|{self.end_numsteps}]')

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def add_ep(self):
        self.total_ep += 1
        return self.total_ep

    def add_total_numsteps(self):
        self.total_numsteps += 1

    def add_train_info(self,
                       critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha,
                       ):
        self.buffer_critic_1_loss.append(critic_1_loss)
        self.buffer_critic_2_loss.append(critic_2_loss)
        self.buffer_policy_loss.append(policy_loss)
        self.buffer_ent_loss.append(ent_loss)
        self.buffer_alpha.append(alpha)

        if not os.path.isfile('./DB/TRAIN_INFO/info.txt'):
            with open('./DB/TRAIN_INFO/info.txt', 'w') as f:
                f.write(f"{'critic_1_loss':30},{'critic_2_loss':30},{'policy_loss':30},{'ent_loss':30},{'alpha':30}\n")
        with open('./DB/TRAIN_INFO/info.txt', 'a') as f:
            f.write(f"{critic_1_loss:30},{critic_2_loss:30},{policy_loss:30},{ent_loss:30},{alpha:30}\n")


    def get_len(self):
        return len(self.buffer)

    def get_ep(self):
        return self.total_ep

    def get_total_numstps(self):
        return self.total_numsteps

    def get_train_info(self):
        return self.buffer_critic_1_loss, self.buffer_critic_2_loss,\
               self.buffer_policy_loss, self.buffer_ent_loss, self.buffer_alpha

    def get_finish_info(self):
        finish = True if self.total_numsteps > self.end_numsteps else False
        return finish