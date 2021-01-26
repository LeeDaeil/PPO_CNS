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

        self.buffer_acc_reward = []

        self.position = 0

        self.total_ep = 0

        self.total_numsteps = 0

        self.end_numsteps = 100

        print(f'[{"Replay Memory Info":20}][Capacity|{self.capacity}][End_Stes|{self.end_numsteps}]')

    def push(self, state, action, reward, next_state, done):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return: No return
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size=int):
        """
        batch size 만큼 데이터를 샘플링 함.
        :param batch_size:
        :return:
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def add_ep(self):
        """
        전체 episode 수를 증가 시키고 그 값을 반환
        :return:
        """
        self.total_ep += 1
        return self.total_ep

    def add_total_numsteps(self):
        """
        전체 step 수를 증가 시키고 그 값을 반환
        :return:
        """
        self.total_numsteps += 1
        return self.total_numsteps

    def add_train_info(self,
                       critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha
                       ):
        """
        loss 값을 loss 저장소에 저장함.
        :param critic_1_loss:
        :param critic_2_loss:
        :param policy_loss:
        :param ent_loss:
        :param alpha:
        :return: No return
        """
        self.buffer_critic_1_loss.append(critic_1_loss)
        self.buffer_critic_2_loss.append(critic_2_loss)
        self.buffer_policy_loss.append(policy_loss)
        self.buffer_ent_loss.append(ent_loss)
        self.buffer_alpha.append(alpha)

        if not os.path.isfile('./DB/TRAIN_INFO/info.txt'):
            with open('./DB/TRAIN_INFO/info.txt', 'w') as f:
                f.write(f"{'critic_1_loss':30},{'critic_2_loss':30},{'policy_loss':30},"
                        f"{'ent_loss':30},{'alpha':30}\n")
        with open('./DB/TRAIN_INFO/info.txt', 'a') as f:
            f.write(f"{critic_1_loss:30},{critic_2_loss:30},{policy_loss:30},{ent_loss:30},{alpha:30}\n")

    def add_ep_end_info(self, acc_reward):
        """

        :param acc_reward:
        :return:
        """
        self.buffer_acc_reward.append(acc_reward)

    def get_len(self):
        return len(self.buffer)

    def get_ep(self):
        return self.total_ep

    def get_total_numstps(self):
        return self.total_numsteps

    def get_train_info(self):
        return self.buffer_critic_1_loss, self.buffer_critic_2_loss,\
               self.buffer_policy_loss, self.buffer_ent_loss, self.buffer_alpha

    def get_ep_end_info(self):
        return self.buffer_acc_reward

    def get_finish_info(self):
        finish = True if self.total_numsteps > self.end_numsteps else False
        return finish