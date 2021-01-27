from Model_0_Basic.ENVS.CNS_Basic import CNS
from Model_2_Start_up_PZR.ENVS.PID_Na import PID

import numpy as np
import time
import random
import copy


class CMem:
    def __init__(self, mem):
        self.m = mem  # Line CNSmem -> getmem
        self.update()

    def update(self):
        self.CTIME = self.m['KCNTOMS']['Val']  # CNS Time
        self.CDelt = self.m['TDELTA']['Val']
        self.PZRPres = self.m['ZINST65']['Val']
        self.PZRLevl = self.m['ZINST63']['Val']
        self.PZRTemp = self.m['UPRZ']['Val']
        self.ExitCoreT = self.m['UUPPPL']['Val']

        self.FV122 = self.m['BFV122']['Val']
        self.FV122M = self.m['KLAMPO95']['Val']

        self.HV142 = self.m['BHV142']['Val']
        self.HV142Flow = self.m['WRHRCVC']['Val']

        self.PZRSprayPos = self.m['ZINST66']['Val']

        self.LetdownSet = self.m['ZINST36']['Val']  # Letdown setpoint
        self.LetdownSetM = self.m['KLAMPO89']['Val']  # Letdown setpoint Man0/Auto1

        self.PZRBackUp = self.m['KLAMPO118']['Val']
        self.PZRPropo = self.m['QPRZH']['Val']

        # Logic


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
        # --------------------------------------------------------------------------------------------------------------
        # 추가된 내용
        self.ActLoggerPath = 'DB/CNS_EP_ENV'
        # RL -----------------------------------------------------------------------------------------------------------
        self.input_info = [
            # (para, x_round, x_min, x_max), (x_min=0, x_max=0 is not normalized.)
            ('BHV142',     1, 0,   0),       # Letdown(HV142)
            # ('WRHRCVC',    1, 0,   0),       # RHR to CVCS Flow
            # ('WNETLD',     1, 0,   10),      # Total Letdown Flow
            # ('BFV122',     1, 0,   0),       # ChargingValve(FV122)
            # ('WNETCH',     1, 0,   10),      # Total Charging Flow
            ('ZINST66',    1, 0,   30),      # PZR spray
            ('ZINST65',    1, 0,   160),     # RCSPressure
            ('ZINST63',    1, 0,   100),     # PZRLevel
            # ('UUPPPL',     1, 0,   200),     # Core Exit Temperature
            # ('UPRZ',       1, 0,   300),     # PZR Temperature
            # ('ZINST36',  1, 0,   0),      # Letdown Pressrue
            # ('SetPres',    1, 0,   100),      # Pres-Setpoint
            # ('SetLevel',   1, 0,   30),      # Level-Setpoint
            # ('ErrPres',    1, 0,   100),     # RCSPressure - setpoint
            # ('UpPres',     1, 0,   100),     # RCSPressure - Up
            # ('DownPres',   1, 0,   100),     # RCSPressure - Down
            # ('ErrLevel',   1, 0,   100),     # PZRLevel - setpoint
            # ('UpLevel',    1, 0,   100),     # PZRLevel - Up
            # ('DownLevel',  1, 0,   100),     # PZRLevel - Down
        ]

        self.action_space = 2       # HV142, Spray, / Charging Valve
        self.observation_space = len(self.input_info) * self.time_leg
        # -------------------------------------------------------------------------------------
        # PID Part
        self.PID_Mode = False
        self.PID_Prs = PID(kp=0.03, ki=0.001, kd=1.0)
        self.PID_Prs_S = PID(kp=0.03, ki=0.001, kd=1.0)
        self.PID_Lev = PID(kp=0.03, ki=0.001, kd=1.0)
        self.PID_Prs.SetPoint = 27.0  # Press set-point
        self.PID_Prs_S.SetPoint = 27.0  # Press set-point
        self.PID_Lev.SetPoint = 30.0  # Level set-point
        # -------------------------------------------------------------------------------------

    # ENV TOOLs ========================================================================================================
    def ENVlogging(self, s):
        cr_time = time.strftime('%c', time.localtime(time.time()))

        if s == '':
            with open(f'./{self.ENVLoggerPath}/{self.Name}.txt', 'a') as f:
                f.write(f'[{cr_time}]|[{self.Mother_current_ep:06}][Start]\n')

            with open(f'./{self.ActLoggerPath}/Act_{self.Name}.txt', 'a') as f:
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
                state.append(self.normalize(self.mem[para]['Val'], x_round, x_min, x_max))
            else:
                # ADD logic ----- 계산된 값을 사용하고 싶을 때
                pass

        return np.array(state), state

    def get_reward(self, A):
        """
        :param A: tanh (-1 ~ 1) 사이 값
        :return:
        """
        r, r_1, r_2, r_3, r_4, r_5 = 0, 0, 0, 0, 0, 0
        # --------------------------------------------------------------------------------------------------------------
        # r_1] Cooling Rate 에 따라 온도 감소
        def get_distance_r(curent_val, set_val, max_val, distance_normal):
            r = 0
            if curent_val - set_val == 0:
                r += max_val
            else:
                if curent_val > set_val:
                    r += (distance_normal - (curent_val - set_val)) / distance_normal
                else:
                    r += (distance_normal - (- curent_val + set_val)) / distance_normal
            r = np.clip(r, 0, max_val)
            return r

        if self.CMem.PZRLevl >= 40:  # 기포 생성 이전
            # 압력
            r_1 += get_distance_r(self.CMem.PZRPres, self.PID_Prs.SetPoint, max_val=1, distance_normal=10)
            # 수위
            r_2 += get_distance_r(self.CMem.PZRLevl, self.PID_Lev.SetPoint, max_val=1, distance_normal=70) / 20
            # 제어
            if A[0] == 0: r_3 += 0.1
        else:  # 기포 생성 이후
            # 압력
            r_1 += get_distance_r(self.CMem.PZRPres, self.PID_Prs.SetPoint, max_val=1, distance_normal=10)
            # 수위
            r_2 += get_distance_r(self.CMem.PZRLevl, self.PID_Lev.SetPoint, max_val=1, distance_normal=70) / 20
            # 제어
            # if abs(A[0]) < 0.6 and abs(A[1]) < 0.6: c+= 0.05
            # 단계적 목표

        r_w = [1 * r_1, 1 * r_2, 1 * r_3, 0 * r_4, 0 * r_5]
        r = sum(r_w)
        # --------------------------------------------------------------------------------------------------------------
        self.Loger_txt += f'R|{r}|{r_1}|{r_2}|{r_3}|{r_4}|{r_5}|{r_w}|'
        return r

    def get_done(self, r, AMod):
        cond = {
            1: True if 176 < self.CMem.ExitCoreT else False,

            2: True if abs(self.CMem.PZRPres - self.PID_Prs.SetPoint) >= 10 else False,  # 목표 set-point 보다 10만큼 크면 종료
            3: True if self.CMem.PZRLevl <= 25 else False,
            4: True if self.CMem.CTIME > 550 * 50 and self.CMem.PZRLevl > 98 else False,
            5: True if 28 <= self.CMem.PZRLevl <= 32 else False,
            6: True if 28 <= self.CMem.PZRPres <= 32 else False,
            7: True if self.CMem.PZRLevl >= 99 else False,
        }
        d = False
        # --------------------------------------------------------------------------------------------------------------
        # 1] 불만족시 즉각 Done
        if cond[1]:
            # 에피소드 종료 조건 도달
            if cond[5] and cond[6]:
                # 에피소드 종료 조건에 도달하였고, 목표하는 범위내에 존재하는 경우 done 은 false 이므로 이를 고려하여 mask 설계
                pass
            else:
                # 에피소드의 종료 조건에 도달하였으나, 목표하는 범위내에 존재하지 않음.
                d = True
        else:
            # 에피소드 진행중 ..
            if cond[7]:
                # 가압기 내 기포가 생성되지 않은 상태임.
                if -0.3 <= AMod[1] < 0.3:
                    pass
                else:
                    d = True
            else:
                # 가압기 내 기포가 생성되었으며, 이제 2번째 액션이 허가됨.
                if cond[2] or cond[3] or cond[4]:
                    # 이중 하나라도 걸리면 에피소드 종료
                    d = True
                else:
                    # 계속 진행 ...
                    pass
        # --------------------------------------------------------------------------------------------------------------
        if d: print(cond)
        # --------------------------------------------------------------------------------------------------------------
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
        :param A: A 액션 [0, 0, 0] <- act space 에 따라서
        :return: AMod: 수정된 액션
        """
        AMod = A
        ActOrderBook = {
            # Charging Valve
            'ChargingValveMan': (['KSWO100'], [1]), 'ChargingValveAUto': (['KSWO100'], [0]),
            'ChargingValveDown': (['KSWO101', 'KSWO102'], [1, 0]),
            'ChargingValveStay': (['KSWO101', 'KSWO102'], [0, 0]),
            'ChargingValveUp': (['KSWO101', 'KSWO102'], [0, 1]),
            'ChargingEdit': (['BFV122'], [0.12]),
            # LetDown Valve
            'LetdownValveOpen': (['KSWO231', 'KSWO232'], [0, 1]),
            'LetdownValveStay': (['KSWO231', 'KSWO232'], [0, 0]),
            'LetdownValveClose': (['KSWO231', 'KSWO232'], [1, 0]),

            'LetdownPresSetUp': (['KSWO90', 'KSWO91'], [0, 1]),
            'LetdownPresSetStay': (['KSWO90', 'KSWO91'], [0, 1]),
            'LetdownPresSetDown': (['KSWO90', 'KSWO91'], [1, 0]),
            'LetdownPresSetA': (['KSWO89'], [0]),

            'LetDownSetDown': (['KSWO90', 'KSWO91'], [1, 0]),
            'LetDownSetStay': (['KSWO90', 'KSWO91'], [0, 0]),
            'LetDownSetUP': (['KSWO90', 'KSWO91'], [0, 1]),
            # PZR Backup/Proportion Heater
            'PZRBackHeaterOff': (['KSWO125'], [0]), 'PZRBackHeaterOn': (['KSWO125'], [1]),
            'PZRProHeaterMan': (['KSWO120'], [1]), 'PZRProHeaterAuto': (['KSWO120'], [0]),
            'PZRProHeaterDown': (['KSWO121', 'KSWO122'], [1, 0]),
            'PZRProHeaterStay': (['KSWO121', 'KSWO122'], [0, 0]),
            'PZRProHeaterUp': (['KSWO121', 'KSWO122'], [0, 1]),
            # PZR Spray
            'PZRSprayMan': (['KSWO128'], [1]), 'PZRSprayAuto': (['KSWO128'], [0]),
            'PZRSprayOpen': (['KSWO126', 'KSWO127'], [0, 1]),
            # 'PZRSprayOpen': (['BPRZSP'], [self.mem['BPRZSP']['Val'] + 0.015 * 1]),
            'PZRSprayStay': (['KSWO126', 'KSWO127'], [0, 0]),
            # 'PZRSprayClose': (['BPRZSP'], [self.mem['BPRZSP']['Val'] + 0.015 * -1]),
            'PZRSprayClose': (['KSWO126', 'KSWO127'], [1, 0]),
            # Delta Time
            'ChangeDelta': (['TDELTA'], [1.0]),

            # # ETC
            # 'StopAllRCP': (['KSWO132', 'KSWO133', 'KSWO134'], [0, 0, 0]),
            # 'StopRCP1': (['KSWO132'], [0]),
            # 'StopRCP2': (['KSWO133'], [0]),
            # 'StopRCP3': (['KSWO134'], [0]),
            # 'NetBRKOpen': (['KSWO244'], [0]),
            # 'OilSysOff': (['KSWO190'], [0]),
            # 'TurningGearOff': (['KSWO191'], [0]),
            # 'CutBHV311': (['BHV311', 'FKAFWPI'], [0, 0]),
            #
            # 'SteamDumpMan': (['KSWO176'], [1]), 'SteamDumpAuto': (['KSWO176'], [0]),
            #
            # 'IFLOGIC_SteamDumpUp': (['PMSS'], [self.CMem.PMSS + 2.0E5 * 3 * 0.2]),
            # 'IFLOGIC_SteamDumpDown': (['PMSS'], [self.CMem.PMSS + 2.0E5 * (-3) * 0.2]),
            #
            # 'DecreaseAux1Flow': (['KSWO142', 'KSWO143'], [1, 0]),
            # 'IncreaseAux1Flow': (['KSWO142', 'KSWO143'], [0, 1]),
            # 'DecreaseAux2Flow': (['KSWO151', 'KSWO152'], [1, 0]),
            # 'IncreaseAux2Flow': (['KSWO151', 'KSWO152'], [0, 1]),
            # 'DecreaseAux3Flow': (['KSWO154', 'KSWO155'], [1, 0]),
            # 'IncreaseAux3Flow': (['KSWO154', 'KSWO155'], [0, 1]),
            #
            # 'SteamLine1Open': (['KSWO148', 'KSWO149'], [1, 0]),
            # 'SteamLine2Open': (['KSWO146', 'KSWO147'], [1, 0]),
            # 'SteamLine3Open': (['KSWO144', 'KSWO145'], [1, 0]),
            #
            # 'ResetSI': (['KSWO7', 'KSWO5'], [1, 1]),
            #
            # 'RL_IncreaseAux1Flow': (['WAFWS1'], [self.mem['WAFWS1']['Val'] + 0.04 * 1]),
            # 'RL_DecreaseAux1Flow': (['WAFWS1'], [self.mem['WAFWS1']['Val'] + 0.04 * (-1)]),
            # 'RL_IncreaseAux2Flow': (['WAFWS2'], [self.mem['WAFWS2']['Val'] + 0.04 * 1]),
            # 'RL_DecreaseAux2Flow': (['WAFWS2'], [self.mem['WAFWS2']['Val'] + 0.04 * (-1)]),
            # 'RL_IncreaseAux3Flow': (['WAFWS3'], [self.mem['WAFWS3']['Val'] + 0.04 * 1]),
            # 'RL_DecreaseAux3Flow': (['WAFWS3'], [self.mem['WAFWS3']['Val'] + 0.04 * (-1)]),
            # #
            # 'RunRCP2': (['KSWO130', 'KSWO133'], [1, 1]),
            # 'RunCHP2': (['KSWO70'], [1]), 'StopCHP2': (['KSWO70'], [0]),
            # 'OpenSI': (['KSWO81', 'KSWO82'], [1, 0]), 'CloseSI': (['KSWO81', 'KSWO82'], [0, 1]),

        }
        # Action Logger ------------------------------------------------------------------------------------------------
        cr_time = time.strftime('%c', time.localtime(time.time()))
        self.a_log = ''

        def a_log_f(s=''):
            self.a_log += f'[{cr_time}]|[{self.Mother_current_ep:06}]\t|{s}\n'
        # --------------------------------------------------------------------------------------------------------------
        a_log_f(s=f'[Step|{self.ENVStep:10}][{"="*20}]')

        # 0] Delta time 빠르게 ...
        if self.CMem.CDelt != 1:
            self._send_control_save(ActOrderBook['ChangeDelta'])
            a_log_f(s=f'ChangeDelta [{self.CMem.CDelt}]')
        # 1] BackUp/Proportion Heater On
        if self.CMem.PZRBackUp != 0:
            self._send_control_save(ActOrderBook['PZRBackHeaterOn'])
            a_log_f(s=f'PZR Backup [{self.CMem.PZRBackUp}] On')
        if self.CMem.PZRPropo != 1:
            self._send_control_save(ActOrderBook['PZRProHeaterUp'])
            a_log_f(s=f'PZR Proportion heater [{self.CMem.PZRPropo}] Increase')

        # 2] Core
        if self.CMem.PZRLevl >= 99:  # 가압기 기포 생성 이전
            self.PID_Prs.SetPoint = 27
            # ----------------------------- PRESS -------------------------------------------------
            if self.PID_Mode:
                PID_out = self.PID_Prs.update(self.CMem.PZRPres, 1)
                if PID_out >= 0.005:
                    self._send_control_save(ActOrderBook['LetdownValveClose'])
                elif -0.005 < PID_out < 0.005:
                    self._send_control_save(ActOrderBook['LetdownValveStay'])
                else:
                    self._send_control_save(ActOrderBook['LetdownValveOpen'])
            else:
                # A[0] HV142 / Spray
                if AMod[0] < -0.3:
                    # Decrease
                    self._send_control_save(ActOrderBook['LetdownValveClose'])
                    a_log_f(s=f'[{"Bubble X":10}]LetdownValveClose')
                elif -0.3 <= AMod[0] < 0.3:
                    # Stay
                    self._send_control_save(ActOrderBook['LetdownValveStay'])
                    a_log_f(s=f'[{"Bubble X":10}]LetdownValveStay')
                elif 0.3 <= AMod[0]:
                    # Increase
                    self._send_control_save(ActOrderBook['LetdownValveOpen'])
                    a_log_f(s=f'[{"Bubble X":10}]LetdownValveOpen')

            # ----------------------------- Level -------------------------------------------------
            if self.PID_Mode:
                PID_out = self.PID_Lev.update(self.CMem.PZRLevl, 1)
                # if PID_out >= 0.005:
                #     self._send_control_save(ActOrderBook['ChargingValveOpen'])
                # elif -0.005 < PID_out < 0.005:
                #     self._send_control_save(ActOrderBook['ChargingValveStay'])
                # else:
                #     self._send_control_save(ActOrderBook['ChargingValveClose'])
            else:
                # A[1] FV122
                pass
            # ----------------------------- ----- -------------------------------------------------
        else:  # 가압기 기포 생성 이후
            self.PID_Prs.SetPoint = 30
            self.PID_Prs_S.SetPoint = 30
            self.PID_Lev.SetPoint = 30
            # ----------------------------- PRESS -------------------------------------------------
            if self.PID_Mode:
                # HV142 ----------------------------------------------------------
                PID_out = self.PID_Prs.update(self.CMem.PZRPres, 1)

                if self.CMem.HV142 != 0:
                    self._send_control_save(ActOrderBook['LetdownValveClose'])
                    a_log_f(s=f'[{"Bubble O":10}]LetdownValveClose')

                # if PID_out >= 0.005:
                #     self._send_control_save(ActOrderBook['LetdownValveClose'])
                # elif -0.005 < PID_out < 0.005:
                #     self._send_control_save(ActOrderBook['LetdownValveStay'])
                # else:
                #     self._send_control_save(ActOrderBook['LetdownValveOpen'])

                # Spray ----------------------------------------------------------
                PID_out = self.PID_Prs_S.update(self.CMem.PZRPres, 1)
                if PID_out >= 0.005:
                    self._send_control_save(ActOrderBook['PZRSprayClose'])
                elif -0.005 < PID_out < 0.005:
                    self._send_control_save(ActOrderBook['PZRSprayStay'])
                else:
                    self._send_control_save(ActOrderBook['PZRSprayOpen'])
                # LetPress ----------------------------------------------------------
                if self.CMem.LetdownSetM == 1:
                    self._send_control_save(ActOrderBook['LetdownPresSetA'])

                # print(f'GetPoint|{self.CMem.PZRPres}, {self.CMem.PZRPres}|\n'
                #       f'LetdownPos:{self.CMem.HV142}:{self.CMem.HV142Flow}|'
                #       f'PZRSpray:{self.CMem.PZRSprayPos}|{PID_out}')
            else:
                # HV142 ----------------------------------------------------------
                # A[0] HV142
                # AMod[0] = -1
                if self.CMem.HV142 != 0:
                    self._send_control_save(ActOrderBook['LetdownValveClose'])
                    a_log_f(s=f'[{"Bubble O":10}]LetdownValveClose')

                # Spray ----------------------------------------------------------
                # A[0] HV142 / Spray
                if AMod[0] < -0.3:
                    # Decrease
                    self._send_control_save(ActOrderBook['PZRSprayClose'])
                    a_log_f(s=f'[{"Bubble X":10}]PZRSprayClose')
                elif -0.3 <= AMod[0] < 0.3:
                    # Stay
                    self._send_control_save(ActOrderBook['PZRSprayStay'])
                    a_log_f(s=f'[{"Bubble X":10}]PZRSprayStay')
                elif 0.3 <= AMod[0]:
                    # Increase
                    self._send_control_save(ActOrderBook['PZRSprayOpen'])
                    a_log_f(s=f'[{"Bubble X":10}]PZRSprayOpen')

                # LetPress ----------------------------------------------------------
                if self.CMem.LetdownSetM == 1:
                    self._send_control_save(ActOrderBook['LetdownPresSetA'])
                    a_log_f(s=f'[{"Bubble X":10}]LetdownPresSetA')

            # ----------------------------- Level -------------------------------------------------
            if self.PID_Mode:
                PID_out = self.PID_Lev.update(self.CMem.PZRLevl, 1)
                if PID_out >= 0.005:
                    self._send_control_save(ActOrderBook['ChargingValveOpen'])
                elif -0.005 < PID_out < 0.005:
                    self._send_control_save(ActOrderBook['ChargingValveStay'])
                else:
                    self._send_control_save(ActOrderBook['ChargingValveClose'])
            else:
                # A[2] FV122
                if AMod[1] < -0.3:
                    # Decrease
                    self._send_control_save(ActOrderBook['ChargingValveClose'])
                    a_log_f(s=f'[{"Bubble O":10}]ChargingValveClose')
                elif -0.3 <= AMod[1] < 0.3:
                    # Stay
                    self._send_control_save(ActOrderBook['ChargingValveStay'])
                    a_log_f(s=f'[{"Bubble O":10}]ChargingValveStay')
                elif 0.3 <= AMod[1]:
                    # Increase
                    self._send_control_save(ActOrderBook['ChargingValveOpen'])
                    a_log_f(s=f'[{"Bubble O":10}]PZRSprayOpen')

        # Action Logger Save -------------------------------------------------------------------------------------------
        with open(f'./{self.ActLoggerPath}/Act_{self.Name}.txt', 'a') as f:
            # Action log
            f.write(f'{self.a_log}')
        # Done Act -----------------------------------------------------------------------------------------------------
        self._send_control_to_cns()
        return AMod

    # ENV Main TOOLs ===================================================================================================
    def step(self, A):
        """
        A를 받고 1 step 전진
        :param A: [Act], numpy.ndarry, Act 는 numpy.float32
        :return: 최신 state 와 reward done 반환
        """
        # Old Data (time t) ---------------------------------------
        self.Loger_txt += f'S|{self.state_txt}|'                            # [s(t)]
        AMod = self.send_act(A)                                             # [a(t)]
        self.Loger_txt += f'A|{A}|AMod|{AMod}'                              #

        self.want_tick = int(50)

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
        # 2] 업데이트된 'Val' 를 'List' 에 추가 및 ENVLogging 초기화
        self._append_val_to_list()
        # 3] 'Val' 을 상태로 제작후 반환
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
                a = 0
                max_iter += 1
                next_state, reward, done, amod = self.env.step(a, std_=1, mean_=0)
                print(f'Doo--{start}->{time.time()} [{time.time() - start}]')
                if done or max_iter >= 2000:
                    print(f'END--{start}->{time.time()} [{time.time() - start}]')
                    break


if __name__ == '__main__':
    Model = CNSTestEnv()
    Model.run_()
