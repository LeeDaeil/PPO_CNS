from Model_0_Basic.AGENT.SAC_init import SAC_Base
from Model_0_Basic.AGENT.Utils import hard_update, soft_update, ensure_shared_grads

from Model_2_Start_up_PZR.AGENT.Networks import GaussianPolicy, QNetwork, DeterministicPolicy

import os
import torch as T
import torch.nn.functional as F
from torch.optim import Adam
import random
import time


class SAC(SAC_Base):
    def __init__(self,
                 nub_agent,
                 env, replay_buffer,
                 lr, policy_type, automatic_entropy_tuning,
                 sh_net=None,
                 ):
        self.env = env
        self.replay_buffer = replay_buffer
        super(SAC, self).__init__(action_space=env.action_space,
                                  observation_space=env.observation_space, replay_buffer=replay_buffer,
                                  lr=lr, policy_type=policy_type, automatic_entropy_tuning=automatic_entropy_tuning)
        self.nub_agent = nub_agent
        self.p_info = f'[{"SAC " + f"{nub_agent}":20}][InCode]'
        print(self.p_info + f'[Env_id|{self.env}][ReplayBuffer|{self.replay_buffer}]')
        # ==============================================================================================================
        #
        # Agent --------------------------------------------------------------------------------------------------------
        if not sh_net == None:
            # shared net -----------------------------------------------------------------------------------------------
            print(self.p_info + f'[Env_id|{self.env}][Weight share !!]')
            # Agent Critic ---------------------------------------------------------------------------------------------
            self.sh_net = sh_net
            self.critic = QNetwork(self.env.observation_space, self.env.action_space, hidden_dim=256)
            self.critic_optim = self.sh_net['critic_opt']
            self.critic_target = QNetwork(self.env.action_space, self.env.observation_space, hidden_dim=256)

            self.critic.load_state_dict(self.sh_net['critic'].state_dict())
            self.critic_target.load_state_dict(self.sh_net['target'].state_dict())

            # Agent Policy ---------------------------------------------------------------------------------------------
            if self.policy_type == "Gaussian":
                self.alpha = 0.2
                self.automatic_entropy_tuning = False

                self.policy = GaussianPolicy(self.env.observation_space, self.env.action_space, 256)
                self.policy_optim = self.sh_net['policy_opt']

                self.policy.load_state_dict(self.sh_net['policy'].state_dict())
            else:
                self.alpha = 0
                self.automatic_entropy_tuning = False

                self.policy = DeterministicPolicy(self.env.observation_space, self.env.action_space, 256)
                self.policy_optim = self.sh_net['policy_opt']

                self.policy.load_state_dict(self.sh_net['policy'].state_dict())
        else:
            # not shared net -------------------------------------------------------------------------------------------
            print(self.p_info + f'[Env_id|{self.env}][No share !!]')
            # Agent Critic ---------------------------------------------------------------------------------------------
            self.critic = QNetwork(self.env.observation_space, self.env.action_space, hidden_dim=256)
            self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
            self.critic_target = QNetwork(self.env.action_space, self.env.observation_space, hidden_dim=256)
            hard_update(self.critic_target, self.critic)
            # Agent Policy ---------------------------------------------------------------------------------------------
            if self.policy_type == "Gaussian":
                self.alpha = 0.2
                self.automatic_entropy_tuning = False

                self.policy = GaussianPolicy(self.env.observation_space, self.env.action_space, 256)
                self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
            else:
                self.alpha = 0
                self.automatic_entropy_tuning = False

                self.policy = DeterministicPolicy(self.env.observation_space, self.env.action_space, 256)
                self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        # Old .. 아직 automatic_entropy_tuning 구현 x
        # if self.policy_type == "Gaussian":
        #     # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        #     if self.automatic_entropy_tuning is True:
        #         self.target_entropy = - T.prod(T.Tensor((self.env.action_space,))).item()
        #         self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        #         self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        #     self.policy = GaussianPolicy(self.env.observation_space, self.env.action_space, 256)
        # else:
        #     self.alpha = 0
        #     self.automatic_entropy_tuning = False
        #     self.policy = DeterministicPolicy(self.env.observation_space, self.env.action_space, 256)
        # self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        # End Agent code -----------------------------------------------------------------------------------------------
        #
        # ==============================================================================================================
        self.run()

    def agent_select_action(self, state, evaluate=False):
        state = T.FloatTensor(state).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]     # [ ], numpy[0.80986434 0.7939146 ] <class 'numpy.ndarray'>

    def agent_update_parameters(self, batch_data):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch_data
        #
        # state_batch = T.FloatTensor(state_batch)
        # next_state_batch = T.FloatTensor(next_state_batch)
        # action_batch = T.FloatTensor(action_batch)
        # reward_batch = T.FloatTensor(reward_batch).unsqueeze(1)
        # mask_batch = T.FloatTensor(mask_batch).unsqueeze(1)
        #
        # with T.no_grad():
        #     next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
        #     qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
        #     min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        #     next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        #
        # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf_loss = qf1_loss + qf2_loss
        #
        # self.critic_optim.zero_grad()
        # qf_loss.backward()
        # if not self.sh_net == None:
        #     ensure_shared_grads(self.critic, self.sh_net['critic'])
        # self.critic_optim.step()
        #
        # pi, log_pi, _ = self.policy.sample(state_batch)
        #
        # qf1_pi, qf2_pi = self.critic(state_batch, pi)
        # min_qf_pi = T.min(qf1_pi, qf2_pi)
        #
        # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        #
        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        # if not self.sh_net == None:
        #     ensure_shared_grads(self.policy, self.sh_net['policy'])
        # self.policy_optim.step()

        if self.automatic_entropy_tuning:
            # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            #
            # self.alpha_optim.zero_grad()
            # alpha_loss.backward()
            # self.alpha_optim.step()
            #
            # self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone() # For TensorboardX logs

            # if self.replay_buffer.get_total_numstps() % self.update_target_per_step == 0:
            #     soft_update(self.critic_target, self.critic, self.tau)
            #
            # return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
            pass
        else:
            # alpha_loss = T.tensor(0.)
            # alpha_tlogs = T.tensor(self.alpha) # For TensorboardX logs
            #
            # if self.replay_buffer.get_total_numstps() % self.update_target_per_step == 0:
            #     soft_update(self.critic_target, self.critic, self.tau)
            #
            #     if not self.sh_net == None:
            #         ensure_shared_grads(self.critic_target, self.sh_net['target'])

            if not self.sh_net == None:
                self.critic.load_state_dict(self.sh_net['critic'].state_dict())
                self.critic_target.load_state_dict(self.sh_net['target'].state_dict())
                self.policy.load_state_dict(self.sh_net['policy'].state_dict())

            # return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def agent_save_model(self, ep_nub, actor_path=None, critic_path=None):
        if not os.path.exists('./DB/AGENT/'):
            os.makedirs('./DB/AGENT/')
        if actor_path is None:
            actor_path = "./DB/AGENT/sac_actor"
        if critic_path is None:
            critic_path = "./DB/AGENT/sac_critic"
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        T.save(self.policy.state_dict(), actor_path)
        T.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def agent_load_model(self, actor_path="./DB/AGENT/sac_actor", critic_path="./DB/AGENT/sac_critic"):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(T.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(T.load(critic_path))

    def add_para(self, r):
        self.replay_buffer.add_para(env_i=f'{self.nub_agent}',
                                    KCNTOMS=self.env.CMem.CTIME,
                                    UUPPPL=self.env.CMem.ExitCoreT,
                                    UPRZ=self.env.CMem.PZRTemp,
                                    ZINST65=self.env.CMem.PZRPres,
                                    ZINST63=self.env.CMem.PZRLevl,
                                    BHV142=self.env.CMem.HV142,
                                    BFV122=self.env.CMem.FV122,
                                    ZINST66=self.env.CMem.PZRSprayPos,
                                    Reward=r)

    def run(self):
        time.sleep(int(self.nub_agent * 5))

        while True:
            ep_nub = self.replay_buffer.add_ep()
            ep_reward = 0
            ep_steps = 0
            done = False
            state = self.env.reset(file_name=f'{ep_nub}', current_ep=ep_nub)

            while not done:
                # if self.start_step > self.replay_buffer.get_total_numstps():
                #     action = 0  # random
                # else:
                #     action = self.agent_select_action(state)

                # Env interaction --------------------------------------------------------------------------------------
                if self.env.CMem.StartRL == 1:  # 이 이후부터 강화학습 진입.
                    # Env interaction with RL --------------------------------------------------------------------------
                    action = self.agent_select_action(state)

                    if self.replay_buffer.get_len() > self.batch_size:
                        for i in range(self.update_per_step):
                            batch_data = self.replay_buffer.sample(batch_size=self.batch_size)
                            self.agent_update_parameters(batch_data)    # get weight from master

                            # self.replay_buffer.add_train_info(critic_1_loss, critic_2_loss, p_loss, ent_loss, alpha)

                    next_state, reward, done, ep_done, AMod = self.env.step(A=action)
                    self.add_para(reward)

                    # --------------------------------------------------------------------------------------------------
                    ep_steps += 1
                    ep_reward += reward
                    self.replay_buffer.add_total_numsteps()

                    # 종료 조건 섹션 -------------------------------------------------------------------------------------
                    mask = 1 if ep_done else float(not done)
                    # --------------------------------------------------------------------------------------------------
                    self.replay_buffer.push(state, AMod, reward, next_state, mask)
                    state = next_state

                    # print(self.p_info + f'[W][ep_nub|{ep_nub:10}][ep_steps|{ep_steps:10}]'
                    #                     f'[mask|{mask:10}][done|{done:10}]')
                else:
                    # 자동 액션들 수행.
                    # action 이 계산이 되어도 env 에서 액션이 들어가지 않음.
                    action = self.agent_select_action(state)
                    next_state, reward, done, ep_done, AMod = self.env.step(A=action)
                    self.add_para(reward)
                    state = next_state

                # End episode done line
                if self.replay_buffer.get_finish_info(): break

            self.replay_buffer.add_ep_end_info(acc_reward=ep_reward)

            self.replay_buffer.clear_para(f'{self.nub_agent}')
            print('Done--------------------')

            # End worker line
            if self.replay_buffer.get_finish_info(): break
        # --------------------------------------------------------------------------------------------------------------
        print(self.p_info + f'Done Agent Training ... Change Test Mode ...')
        # --------------------------------------------------------------------------------------------------------------
        while True:
            ep_nub = self.replay_buffer.add_ep()
            ep_reward = 0
            ep_steps = 0
            done = False
            state = self.env.reset(file_name=f'{ep_nub}', current_ep=ep_nub)

            while not done:
                # if self.start_step > self.replay_buffer.get_total_numstps():
                #     action = 0  # random
                # else:
                #     action = self.agent_select_action(state)

                # Env interaction --------------------------------------------------------------------------------------
                if self.env.CMem.StartRL == 1:  # 이 이후부터 강화학습 진입.
                    # Env interaction with RL --------------------------------------------------------------------------
                    action = self.agent_select_action(state)

                    if self.replay_buffer.get_len() > self.batch_size:
                        for i in range(self.update_per_step):
                            batch_data = self.replay_buffer.sample(batch_size=self.batch_size)
                            self.agent_update_parameters(batch_data)  # get weight from master

                            # self.replay_buffer.add_train_info(critic_1_loss, critic_2_loss, p_loss, ent_loss, alpha)

                    next_state, reward, done, ep_done, AMod = self.env.step(A=action)
                    self.add_para(reward)

                    # --------------------------------------------------------------------------------------------------
                    ep_steps += 1
                    ep_reward += reward
                    self.replay_buffer.add_total_numsteps()

                    # 종료 조건 섹션 -------------------------------------------------------------------------------------
                    mask = 1 if ep_done else float(not done)
                    # --------------------------------------------------------------------------------------------------
                    # self.replay_buffer.push(state, action, reward, next_state, mask)
                    state = next_state

                    # print(self.p_info + f'[W][ep_nub|{ep_nub:10}][ep_steps|{ep_steps:10}]'
                    #                     f'[mask|{mask:10}][done|{done:10}]')
                else:
                    # 자동 액션들 수행.
                    # action 이 계산이 되어도 env 에서 액션이 들어가지 않음.
                    action = self.agent_select_action(state)
                    next_state, reward, done, ep_done, AMod = self.env.step(A=action)
                    self.add_para(reward)
                    state = next_state

            self.replay_buffer.add_ep_end_info(acc_reward=ep_reward)

            self.replay_buffer.clear_para(f'{self.nub_agent}')
            print('Done--------------------')