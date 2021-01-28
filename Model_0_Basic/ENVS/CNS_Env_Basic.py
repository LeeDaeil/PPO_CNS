from Model_0_Basic.ENVS.CNS_Basic import CNS
import numpy as np
import time
import random


class CMem:
    def __init__(self, mem):
        self.m = mem  # Line CNSmem -> getmem
        self.update()

    def update(self):
        self.CTIME = self.m['KCNTOMS']['Val']       # CNS Time


class ENVCNS(CNS):
    def __init__(self, Name, CNS_IP, CNS_PORT, Remote_IP='192.168.0.29', Remote_PORT=7101):
        super(ENVCNS, self).__init__(threrad_name=Name,
                                     CNS_IP=CNS_IP, CNS_Port=CNS_PORT,
                                     Remote_IP=Remote_IP, Remote_Port=Remote_PORT, Max_len=10)
        # --------------------------------------------------------------------------------------------------------------
        # Initial and Memory
        self.Name = Name  # = id
        self.AcumulatedReward = 0
        self.ENVStep = 0
        self.Mother_current_ep = 0
        self.state_txt = []
        self.LoggerPath = 'DB/CNS_EP_DB'
        self.ENVLoggerPath = 'DB/CNS_EP_ENV'
        self.want_tick = 5  # 1sec
        self.time_leg = 1
        self.Loger_txt = ''
        self.CMem = CMem(self.mem)
        # RL -----------------------------------------------------------------------------------------------------------
        self.input_info = [
            # (para, x_round, x_min, x_max), (x_min=0, x_max=0 is not normalized.)
            ('KCNTOMS',     1, 0,   0),       # Letdown(HV142)
            ('KCNTOMS',     1, 0,   0),       # Letdown(HV142)
        ]

        self.action_space = 2       # TODO HV142 [0], Spray [1], FV122 [2]
        self.observation_space = len(self.input_info) * self.time_leg

    # ENV TOOLs ========================================================================================================
    def ENVlogging(self, s):
        cr_time = time.strftime('%c', time.localtime(time.time()))

        if s == '':
            with open(f'./{self.ENVLoggerPath}/{self.Name}.txt', 'a') as f:
                f.write(f'[{cr_time}]|[{self.Mother_current_ep:06}][Start]\n')
        else:
            with open(f'./{self.ENVLoggerPath}/{self.Name}.txt', 'a') as f:
                f.write(f'[{cr_time}]|[{self.Mother_current_ep:6}][{self.CMem.CTIME:5}]|{self.Loger_txt}\n')

    def normalize(self, x, x_round, x_min, x_max):
        if x_max == 0 and x_min == 0:
            # It means X value is not normalized.
            x = x / x_round
        else:
            x = x_max if x >= x_max else x
            x = x_min if x <= x_min else x
            x = (x - x_min) / (x_max - x_min)
        return x

    # ENV RL TOOLs =====================================================================================================
    def get_state(self):
        state = []
        for para, x_round, x_min, x_max in self.input_info:
            if para in self.mem.keys():
                _ = self.mem[para]['Val']
            else:
                _ = self.PID_Prs.SetPoint if para == 'DSetPoint' else 0
            state.append(self.normalize(_, x_round, x_min, x_max))

        return np.array(state), state

    def get_reward(self, A):
        """

        :param A: tanh (-1 ~ 1) 사이 값
        :return:
        """
        r = 0
        self.Loger_txt += f'R|{r:10}|'
        return r

    def get_done(self, r, AMod):
        d = False

        cond = {
            1: True,
        }

        if cond[1]:
            pass
        else:
            pass

        if d: print(cond)

        self.Loger_txt += f'D|{d}|{cond}|'
        return d, r

    def _send_control_save(self, zipParaVal):
        super(ENVCNS, self)._send_control_save(para=zipParaVal[0], val=zipParaVal[1])

    def send_act(self, A):
        """
        A 에 해당하는 액션을 보내고 나머지는 자동
        E.x)
            self._send_control_save(['KSWO115'], [0])
            ...
            self._send_control_to_cns()
        :param A: A 액션 [0, 0, 0] <- act space에 따라서
        :return: AMod: 수정된 액션
        """
        AMod = A
        ActOrderBook = {
            'ChargingValveOpen': (['KSWO101', 'KSWO102'], [0, 1]),
        }
        # self._send_control_save(ActOrderBook['ChargingValveOpen'])

        #---------------------------------------------------------------------------------------------------------------
        # Done Act

        self._send_control_to_cns()
        return AMod

    # ENV Main TOOLs ===================================================================================================
    def step(self, A):
        """
        A를 받고 1 step 전진
        :param A: [Act], numpy.ndarry, Act는 numpy.float32
        :return: 최신 state와 reward done 반환
        """
        # Old Data (time t) ---------------------------------------
        self.Loger_txt += f'S|{self.state_txt}|'                            # [s(t)]
        AMod = self.send_act(A)                                             # [a(t)]
        self.Loger_txt += f'A|{A}|AMod|{AMod}'                              #
        self.want_tick = int(5)

        # New Data (time t+1) -------------------------------------
        super(ENVCNS, self).step()                  # 전체 CNS mem run-Freeze 하고 mem 업데이트
        self.CMem.update()                          # 선택 변수 mem 업데이트

        self._append_val_to_list()
        self.ENVStep += 1

        reward = self.get_reward(AMod)                                      # [r(t+1)]
        done, reward = self.get_done(reward, AMod)                          # [d(t+1)]
        next_state, next_state_list = self.get_state()                      # [s(t+1)]
        self.Loger_txt += f'NS|{next_state_list}|'                          #
        # ----------------------------------------------------------
        self.ENVlogging(s=self.Loger_txt)
        self.state_txt = next_state_list                                    # [s(t) <- s(t+1)]
        self.Loger_txt = ''
        return next_state, reward, done, AMod

    def reset(self, file_name, current_ep):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=21, mal=False, mal_case=0, mal_opt=0, mal_time=0, file_name=file_name)
        # 2] 업데이트된 'Val'를 'List'에 추가 및 ENVLogging 초기화
        self._append_val_to_list()
        # 3] 'Val'을 상태로 제작후 반환
        self.Mother_current_ep = current_ep
        state, self.state_txt = self.get_state()
        self.ENVlogging(s='')
        self.CMem.update()
        # 4] 보상 누적치 및 ENVStep 초기화
        self.AcumulatedReward = 0
        self.ENVStep = 0
        # 5] FIX RADVAL
        self.FixedRad = random.randint(0, 20) * 5
        self.FixedTime = 0
        self.FixedTemp = 0

        return state


class CNSTestEnv:
    def __init__(self):
        self.env = ENVCNS(Name='Env1', IP='192.168.0.101', PORT=int(f'7101'))

    def run_(self, iter_=1):
        for i in range(1, iter_+1):  # iter = 1 이면 1번 동작
            self.env.reset(file_name=f'Ep{i}')
            start = time.time()
            max_iter = 0
            while True:
                A = 0
                max_iter += 1
                next_state, reward, done, AMod = self.env.step(A, std_=1, mean_=0)
                print(f'Doo--{start}->{time.time()} [{time.time() - start}]')
                if done or max_iter >= 2000:
                    print(f'END--{start}->{time.time()} [{time.time() - start}]')
                    break


if __name__ == '__main__':
    Model = CNSTestEnv()
    Model.run_()