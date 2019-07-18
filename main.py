import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 50 # 몇 번 액션을 취하고 훈련을 수행 할 것인지 결정하는 변수


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(8, 125)
        self.fc2 = nn.Linear(125, 256)
        self.fc_pi = nn.Linear(256, 4)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------
    # 초기 환경에 대한 입력값과 출력값의 갯수를 파악하는 부분
    env = gym.make('LunarLander-v2')

    num_inputs = env.observation_space  # 해당 환경의 입력되는 입력값
    num_actions = env.action_space      # 수행 가능한 Action 의 수
    num_inputs = 8
    num_actions = 4

    # ------------------------------------------------------------------------------------------
    # Actor 와 Critic 모델을 불러오는 부분
    model = PPO()

    # ------------------------------------------------------------------------------------------
    # 에피소드 시작
    score = 0.0
    for n_epi in range(10000):

        # ------------------------------------------------------------------------------------------
        # 환경 초기화 및 첫번째 상태 데이터 가져옴
        s = env.reset()
        done = False
        reward = 0
        # print('첫번째 상태 데이터 : {}'.format(s))

        # ------------------------------------------------------------------------------------------
        # 게임이 종료되기 전까지 계속 훈련 수행
        while not done:

            # ------------------------------------------------------------------------------------------
            # 일정 스텝동안만 환경 탐색
            for t in range(T_horizon):
                if n_epi%100 == 0:
                    env.render()

                # ------------------------------------------------------------------------------------------
                # 현재 상태에 대한 probability 와 action 을 계산
                prob = model.pi(torch.from_numpy(s))
                m = Categorical(prob)
                a = m.sample().item()
                # print('현재 상태에 대한 action 계산 :', prob, m, a)

                # ------------------------------------------------------------------------------------------
                # 도출된 action 에 대한 환경의 반응 취득
                s_prime, r, done, info = env.step(a)

                # ------------------------------------------------------------------------------------------
                # 훈련 데이터 저장
                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))

                s = s_prime

                score += r
                reward += r
                if done:
                    break
                #time.sleep(1)
            model.train_net()

        print(n_epi, score, reward)

    env.close()