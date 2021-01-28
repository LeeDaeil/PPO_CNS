from Model_0_Basic.ENVS.CNS_Basic import CNS

from Model_1_Emergency.ENVS.PTCurve import PTCureve
from Model_1_Emergency.ENVS.CoolingRate import CoolingRATE

import numpy as np
import time
import random
import copy


class CMem:
    def __init__(self, mem):
        self.m = mem  # Line CNSmem -> getmem
        self.CoolingRateSW = 0
        self.CoolingRateFixTemp = 0
        self.CoolingRateFixTime = 0

        self.CoolingRATE = CoolingRATE()

        self.StartRL = 0

        self.update()

    def update(self):
        self.CTIME = self.m['KCNTOMS']['Val']       # CNS Time

        # Physical
        self.SG1Nar = self.m['ZINST78']['Val']
        self.SG2Nar = self.m['ZINST77']['Val']
        self.SG3Nar = self.m['ZINST76']['Val']

        self.SG1Wid = self.m['ZINST72']['Val']
        self.SG2Wid = self.m['ZINST71']['Val']
        self.SG3Wid = self.m['ZINST70']['Val']

        self.SG1Pres = self.m['ZINST75']['Val']
        self.SG2Pres = self.m['ZINST74']['Val']
        self.SG3Pres = self.m['ZINST73']['Val']

        self.SG1Feed = self.m['WFWLN1']['Val']
        self.SG2Feed = self.m['WFWLN2']['Val']
        self.SG3Feed = self.m['WFWLN3']['Val']

        self.Aux1Flow = self.m['WAFWS1']['Val']
        self.Aux2Flow = self.m['WAFWS2']['Val']
        self.Aux3Flow = self.m['WAFWS3']['Val']

        self.SteamLine1 = self.m['BHV108']['Val']
        self.SteamLine2 = self.m['BHV208']['Val']
        self.SteamLine3 = self.m['BHV308']['Val']

        self.AVGTemp = self.m['UAVLEG2']['Val']
        self.PZRPres = self.m['ZINST65']['Val']
        self.PZRLevel = self.m['ZINST63']['Val']

        # Signal
        self.Trip = self.m['KLAMPO9']['Val']
        self.SIS = self.m['KLAMPO6']['Val']
        self.MSI = self.m['KLAMPO3']['Val']
        self.NetBRK = self.m['KLAMPO224']['Val']

        # Comp
        self.RCP1 = self.m['KLAMPO124']['Val']
        self.RCP2 = self.m['KLAMPO125']['Val']
        self.RCP3 = self.m['KLAMPO126']['Val']

        self.TurningGear = self.m['KLAMPO165']['Val']
        self.OilSys = self.m['KLAMPO164']['Val']
        self.BHV311 = self.m['BHV311']['Val']

        self.SteamDumpPos = self.m['ZINST98']['Val']
        self.SteamDumpManAuto = self.m['KLAMPO150']['Val']

        self.PMSS = self.m['PMSS']['Val']

        # 강화학습을 위한 감시 변수
        self.PZRSprayManAuto = self.m['KLAMPO119']['Val']
        self.PZRSprayPos = self.m['ZINST66']['Val']
        self.PZRSprayPosControl = self.m['BPRZSP']['Val']

        self.PZRBackHeaterOnOff = self.m['KLAMPO118']['Val']
        self.PZRProHeaterManAuto = self.m['KLAMPO117']['Val']
        self.PZRProHeaterPos = self.m['BHV22']['Val']

        self.ChargingManAUto = self.m['KLAMPO95']['Val']
        self.ChargingValvePos = self.m['BFV122']['Val']
        self.ChargingPump2State = self.m['KLAMPO70']['Val']

        # Logic
        if self.CTIME == 0:
            self.CoolingRateSW = 0
            self.CoolingRATE.reset_info()

            self.StartRL = 0

        if self.CoolingRateSW == 1:         # 2.0] Cooling rage 계산 시작
            self.CoolingRATE.save_info(self.AVGTemp, self.CTIME)
            self.CoolingRateSW += 1     # 값 2로 바뀜으로써 이 로직은 1번만 동작함.


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
        self.accident_name = ['LOCA', 'SGTR', 'MSLB'][0]
        self.ActLoggerPath = 'DB/CNS_EP_ENV'
        # RL -----------------------------------------------------------------------------------------------------------
        self.input_info = [
            # (para, x_round, x_min, x_max), (x_min=0, x_max=0 is not normalized.)
            ('ZINST98', 1, 0, 100),  # SteamDumpPos
            ('ZINST87', 1, 0, 50),  # Steam Flow 1
            ('ZINST86', 1, 0, 50),  # Steam Flow 2
            ('ZINST85', 1, 0, 50),  # Steam Flow 3
            ('KLAMPO70', 1, 0, 1),  # Charging Pump2 State
            ('BHV22', 1, 0, 1),  # SI Valve State
            ('ZINST66', 1, 0, 25),  # PZRSprayPos
            ('UAVLEG2', 1, 150, 320),  # PTTemp
            ('ZINST65', 1, 0, 160),  # PTPressure
            ('ZINST78', 1, 0, 70),  # SG1Nar
            ('ZINST77', 1, 0, 70),  # SG2Nar
            ('ZINST76', 1, 0, 70),  # SG3Nar
            ('ZINST75', 1, 0, 80),  # SG1Pres
            ('ZINST74', 1, 0, 80),  # SG2Pres
            ('ZINST73', 1, 0, 80),  # SG3Pres
            ('ZINST72', 1, 0, 100),  # SG1Wid
            ('ZINST71', 1, 0, 100),  # SG2Wid
            ('ZINST70', 1, 0, 100),  # SG3Wid
            ('UUPPPL', 1, 100, 350),  # CoreExitTemp
            ('WFWLN1', 1, 0, 25),  # SG1Feed
            ('WFWLN2', 1, 0, 25),  # SG2Feed
            ('WFWLN3', 1, 0, 25),  # SG3Feed
            ('UCOLEG1', 1, 0, 100),  # RCSColdLoop1
            ('UCOLEG2', 1, 0, 100),  # RCSColdLoop2
            ('UCOLEG3', 1, 0, 100),  # RCSColdLoop3
            ('ZINST65', 1, 0, 160),  # RCSPressure
            ('ZINST63', 1, 0, 100),  # PZRLevel
        ]

        self.action_space = 4       # TODO HV142 [0], Spray [1], FV122 [2]
        self.observation_space = len(self.input_info) * self.time_leg

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
        r, r_1, r_2, r_3, r_4, r_5 = 0, 0, 0, 0, 0, 0
        # --------------------------------------------------------------------------------------------------------------
        # r_1] Cooling Rate 에 따라 온도 감소
        if self.CMem.CoolingRateSW == 2:    # Line 99 참조
            current_DRTemp = self.CMem.CoolingRATE.get_temp(self.CMem.CTIME)
            r_1 = - abs(current_DRTemp - self.CMem.AVGTemp)
        # r_2] 가압기 수위 20~76% 구간 초가시 패널티
        if True:
            if 20 <= self.CMem.PZRLevel <= 76:
                pass
            else:
                if 20 > self.CMem.PZRLevel:
                    r_2 -= (20 - self.CMem.PZRLevel)
                else:
                    r_2 -= (self.CMem.PZRLevel - 76)
        # r_3] 증기 발생기 6% ~ 50% 이상 초과 시 패널티
        if True:
            for sg_nar in [self.CMem.SG1Nar, self.CMem.SG2Nar, self.CMem.SG3Nar]:
                if 6 <= sg_nar <= 50:
                    pass
                else:
                    if 6 > sg_nar:
                        r_3 -= (6 - sg_nar)
                    else:
                        r_3 -= (sg_nar - 50)
        # r_4] 목표치와 가까워 질 수록 + 보상
        if True:
            w = [0.1, 0.1]
            r_4 += (29.5 - self.CMem.PZRPres) * w[0] + (170 - self.CMem.AVGTemp) * w[1]
        # r_5] S/G 압력 체크
        if True:
            Avg_pres = (self.CMem.SG1Pres + self.CMem.SG2Pres + self.CMem.SG3Pres) / 3
            r_5 += 9 - Avg_pres if Avg_pres > 9 else 0

        r_w = [1 * r_1, 1 * r_2, 0 * r_3, 0 * r_4, 0 * r_5]
        r = sum(r_w)
        # --------------------------------------------------------------------------------------------------------------
        self.Loger_txt += f'R|{r:10}|{r_1:10}|{r_2:10}|{r_3:10}|{r_4:10}|{r_5:10}|'
        return r

    def get_done(self, r, AMod):
        d = False
        cond = {
            1: True if PTCureve().Check(Temp=self.CMem.AVGTemp, Pres=self.CMem.PZRPres) == 1 else False,  # 0이면 정상
            2: True if self.CMem.PZRPres <= 17 else False,   # 압력이 17이하로 떨어진 것인지
            3: True if max(self.CMem.SG1Nar, self.CMem.SG2Nar, self.CMem.SG3Nar) > 78 else False,  # 증기발생기 수위 78 이상
            4: True if self.CMem.CTIME > 35000 else False,   # 최대 Tick
            5: True if 17 < self.CMem.PZRPres < 29.5 else False,    # 종료 범위 압력
            6: True if self.CMem.AVGTemp <= 170 else False,         # 종료 범위 온도
        }
        # --------------------------------------------------------------------------------------------------------------
        # 1] 불만족시 즉각 Done
        if cond[1] or cond[2] or cond[3]:
            # PT 커브 벗어나거나, 압력이 17이하로 떨어지거나, 증기발생기 수위가 너무 높으면 바로 Done
            d = True
        else:
            if cond[4]:
                if cond[4] and cond[5] and cond[6]:
                    # 최대 시간 도달
                    # !! AGENT에서도 체크 "범위 내에 있으면서 시간 도달하면" done False 이므로 전달함. 이를 고려하여 Mask 설계
                    pass
                else:
                    d = True
            else:
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
            'StopAllRCP': (['KSWO132', 'KSWO133', 'KSWO134'], [0, 0, 0]),
            'StopRCP1': (['KSWO132'], [0]),
            'StopRCP2': (['KSWO133'], [0]),
            'StopRCP3': (['KSWO134'], [0]),
            'NetBRKOpen': (['KSWO244'], [0]),
            'OilSysOff': (['KSWO190'], [0]),
            'TurningGearOff': (['KSWO191'], [0]),
            'CutBHV311': (['BHV311', 'FKAFWPI'], [0, 0]),

            'PZRSprayMan': (['KSWO128'], [1]), 'PZRSprayAuto': (['KSWO128'], [0]),

            'PZRSprayClose': (['BPRZSP'], [self.mem['BPRZSP']['Val'] + 0.015 * -1]),
            'PZRSprayOpen': (['BPRZSP'], [self.mem['BPRZSP']['Val'] + 0.015 * 1]),

            'PZRBackHeaterOff': (['KSWO125'], [0]), 'PZRBackHeaterOn': (['KSWO125'], [1]),

            'SteamDumpMan': (['KSWO176'], [1]), 'SteamDumpAuto': (['KSWO176'], [0]),

            'IFLOGIC_SteamDumpUp': (['PMSS'], [self.CMem.PMSS + 2.0E5 * 3 * 0.2]),
            'IFLOGIC_SteamDumpDown': (['PMSS'], [self.CMem.PMSS + 2.0E5 * (-3) * 0.2]),

            'DecreaseAux1Flow': (['KSWO142', 'KSWO143'], [1, 0]),
            'IncreaseAux1Flow': (['KSWO142', 'KSWO143'], [0, 1]),
            'DecreaseAux2Flow': (['KSWO151', 'KSWO152'], [1, 0]),
            'IncreaseAux2Flow': (['KSWO151', 'KSWO152'], [0, 1]),
            'DecreaseAux3Flow': (['KSWO154', 'KSWO155'], [1, 0]),
            'IncreaseAux3Flow': (['KSWO154', 'KSWO155'], [0, 1]),

            'SteamLine1Open': (['KSWO148', 'KSWO149'], [1, 0]),
            'SteamLine2Open': (['KSWO146', 'KSWO147'], [1, 0]),
            'SteamLine3Open': (['KSWO144', 'KSWO145'], [1, 0]),

            'ResetSI': (['KSWO7', 'KSWO5'], [1, 1]),

            'PZRProHeaterMan': (['KSWO120'], [1]), 'PZRProHeaterAuto': (['KSWO120'], [0]),
            'PZRProHeaterDown': (['KSWO121', 'KSWO122'], [1, 0]),
            'PZRProHeaterUp': (['KSWO121', 'KSWO122'], [0, 1]),

            'RL_IncreaseAux1Flow': (['WAFWS1'], [self.mem['WAFWS1']['Val'] + 0.04 * 1]),
            'RL_DecreaseAux1Flow': (['WAFWS1'], [self.mem['WAFWS1']['Val'] + 0.04 * (-1)]),
            'RL_IncreaseAux2Flow': (['WAFWS2'], [self.mem['WAFWS2']['Val'] + 0.04 * 1]),
            'RL_DecreaseAux2Flow': (['WAFWS2'], [self.mem['WAFWS2']['Val'] + 0.04 * (-1)]),
            'RL_IncreaseAux3Flow': (['WAFWS3'], [self.mem['WAFWS3']['Val'] + 0.04 * 1]),
            'RL_DecreaseAux3Flow': (['WAFWS3'], [self.mem['WAFWS3']['Val'] + 0.04 * (-1)]),

            'ChargingValveMan': (['KSWO100'], [1]), 'ChargingValveAUto': (['KSWO100'], [0]),
            'ChargingValveDown': (['KSWO101', 'KSWO102'], [1, 0]),
            'ChargingValveUp': (['KSWO101', 'KSWO102'], [0, 1]),

            'RunRCP2': (['KSWO130', 'KSWO133'], [1, 1]),
            'RunCHP2': (['KSWO70'], [1]), 'StopCHP2': (['KSWO70'], [0]),
            'OpenSI': (['KSWO81', 'KSWO82'], [1, 0]), 'CloseSI': (['KSWO81', 'KSWO82'], [0, 1]),

        }
        # Action Logger ------------------------------------------------------------------------------------------------
        cr_time = time.strftime('%c', time.localtime(time.time()))
        self.a_log = ''

        def a_log_f(s=''):
            self.a_log += f'[{cr_time}]|[{self.Mother_current_ep:06}]\t|{s}\n'
        # --------------------------------------------------------------------------------------------------------------
        a_log_f(s=f'[Step|{self.ENVStep:10}][{"="*20}]')
        if self.CMem.Trip == 1:
            # 1.1] 원자로 Trip 이후 자동 제어 액션
            # 1.1.1] RCP 97 압력 이하에서 자동 정지
            if self.CMem.RCP1 == 1 and self.CMem.PZRPres < 97 and self.CMem.CTIME < 15 * 60 * 5:
                a_log_f(s=f'Pres [{self.CMem.PZRPres}] < 97 RCP 1 stop')
                self._send_control_save(ActOrderBook['StopRCP1'])
            if self.CMem.RCP2 == 1 and self.CMem.PZRPres < 97 and self.CMem.CTIME < 15 * 60 * 5:
                a_log_f(s=f'Pres [{self.CMem.PZRPres}] < 97 RCP 2 stop')
                self._send_control_save(ActOrderBook['StopRCP2'])
            if self.CMem.RCP3 == 1 and self.CMem.PZRPres < 97 and self.CMem.CTIME < 15 * 60 * 5:
                a_log_f(s=f'Pres [{self.CMem.PZRPres}] < 97 RCP 3 stop')
                self._send_control_save(ActOrderBook['StopRCP3'])
            # 1.1.2] 원자로 트립 후 Netbrk, turning gear, oil sys, BHV311 정지 및 패쇄
            if self.CMem.NetBRK == 1:
                a_log_f(s=f'NetBRK [{self.CMem.NetBRK}] Off')
                self._send_control_save(ActOrderBook['NetBRKOpen'])
            if self.CMem.TurningGear == 1:
                a_log_f(s=f'TurningGear [{self.CMem.TurningGear}] Off')
                self._send_control_save(ActOrderBook['TurningGearOff'])
            if self.CMem.OilSys == 1:
                a_log_f(s=f'OilSys [{self.CMem.OilSys}] Off')
                self._send_control_save(ActOrderBook['OilSysOff'])
            if self.CMem.BHV311 > 0:
                a_log_f(s=f'BHV311 [{self.CMem.BHV311}] Cut')
                self._send_control_save(ActOrderBook['CutBHV311'])
            # 1.2] 스팀 덤프벨브 현재 최대 압력을 기준으로 해당 부분까지 벨브 Set-up
            a_log_f(s=f'[Check][{self.CMem.SIS}][{self.CMem.MSI}][Check Main logic 1]')
            if self.CMem.SIS != 0 and self.CMem.MSI != 0:
                if max(self.CMem.SG1Pres, self.CMem.SG2Pres, self.CMem.SG3Pres) < self.CMem.SteamDumpPos:
                    a_log_f(s=f'StemDumpPos [{self.CMem.SteamDumpPos}] change')
                    self._send_control_save(ActOrderBook['IFLOGIC_SteamDumpDown'])
            # 1.2] SI reset 전에 Aux 평균화 [검증 완료 20200903]
                if self.CMem.SG1Feed == self.CMem.SG2Feed and self.CMem.SG1Feed == self.CMem.SG3Feed and \
                        self.CMem.SG2Feed == self.CMem.SG1Feed and self.CMem.SG2Feed == self.CMem.SG3Feed and \
                        self.CMem.SG3Feed == self.CMem.SG1Feed and self.CMem.SG3Feed == self.CMem.SG2Feed:
                    a_log_f(s=f'[{self.CMem.SG1Feed:10}, {self.CMem.SG2Feed:10}, {self.CMem.SG3Feed:10}] Feed water avg done')
                else:
                    # 1.2.1] 급수 일정화 수행
                    # 1.2.1.1] 가장 큰 급수 찾기
                    SGFeedList = [self.CMem.SG1Feed, self.CMem.SG2Feed, self.CMem.SG3Feed]
                    MaxSGFeed = SGFeedList.index(max(SGFeedList))  # 0, 1, 2
                    MinSGFeed = SGFeedList.index(min(SGFeedList))  # 0, 1, 2
                    self._send_control_save(ActOrderBook[f'DecreaseAux{MaxSGFeed + 1}Flow'])
                    self._send_control_save(ActOrderBook[f'IncreaseAux{MinSGFeed + 1}Flow'])
                    a_log_f(s=f'[{self.CMem.SG1Feed:10}, {self.CMem.SG2Feed:10}, {self.CMem.SG3Feed:10}] Feed water avg')
            # 1.3] 3000부터 SI reset
            if self.CMem.CTIME == 3000:
                self._send_control_save(ActOrderBook['ResetSI'])
                a_log_f(s=f'ResetSI [{self.CMem.CTIME}]')
            # 2] SI reset 발생 시 냉각 운전 시작
            if self.CMem.SIS == 0 and self.CMem.MSI == 0 and self.CMem.CTIME > 5 * 60 * 5:
                # 2.0] Cooling rage 계산 시작
                if self.CMem.CoolingRateSW == 0:
                    self.CMem.CoolingRateSW = 1
                    a_log_f(s=f'CoolingRateSW')
                # 2.1] Press set-point 를 현재 최대 압력 기준까지 조절 ( not work )
                if self.CMem.SteamDumpManAuto == 0:
                    self._send_control_save(ActOrderBook['SteamDumpMan'])
                    a_log_f(s=f'SteamDumpMan [{self.CMem.SteamDumpManAuto}]')
                # 2.2] Steam Line Open
                if self.CMem.SteamLine1 == 0:
                    self._send_control_save(ActOrderBook['SteamLine1Open'])
                    a_log_f(s=f'SteamLine1 [{self.CMem.SteamLine1}] Open')
                if self.CMem.SteamLine2 == 0:
                    self._send_control_save(ActOrderBook['SteamLine2Open'])
                    a_log_f(s=f'SteamLine2 [{self.CMem.SteamLine2}] Open')
                if self.CMem.SteamLine3 == 0:
                    self._send_control_save(ActOrderBook['SteamLine3Open'])
                    a_log_f(s=f'SteamLine3 [{self.CMem.SteamLine3}] Open')
                # 2.3] Charging flow 최소화
                if self.CMem.ChargingManAUto == 0:
                    self._send_control_save(ActOrderBook['ChargingValveMan'])
                    a_log_f(s=f'ChargingMode [{self.CMem.ChargingManAUto}] Man')
                if self.CMem.ChargingValvePos != 0:
                    self._send_control_save(ActOrderBook['ChargingValveDown'])
                    a_log_f(s=f'ChargingPOS [{self.CMem.ChargingValvePos}] Close')
                # 2.2] PZR spray 동작 [감압]
                if self.CMem.PZRSprayManAuto == 0:
                    self._send_control_save(ActOrderBook['PZRSprayMan'])
                    a_log_f(s=f'ChargingPOS [{self.CMem.ChargingValvePos}] Close')
                if self.CMem.RCP2 == 0:
                    self._send_control_save(ActOrderBook['RunRCP2'])
                    a_log_f(s=f'RCP2 [{self.CMem.RCP2}] Start')
                # 2.3] PZR 감압을 위한 Heater 종료
                if self.CMem.PZRProHeaterManAuto == 0:
                    self._send_control_save(ActOrderBook['PZRProHeaterMan'])
                    a_log_f(s=f'PZR PRO heater [{self.CMem.PZRProHeaterManAuto}] Man')
                if self.CMem.PZRProHeaterPos >= 0:
                    self._send_control_save(ActOrderBook['PZRProHeaterDown'])
                    a_log_f(s=f'PZR PRO Pos [{self.CMem.PZRProHeaterPos}] Down')
                if self.CMem.PZRBackHeaterOnOff == 1:
                    self._send_control_save(ActOrderBook['PZRBackHeaterOff'])
                    a_log_f(s=f'PZRBackHeaterOff [{self.CMem.PZRBackHeaterOnOff}] Off')
                # 3.0] 강화학습 제어 시작
                if self.CMem.StartRL == 0:
                    self.CMem.StartRL = 1
                    a_log_f(s=f'StartRL [{self.CMem.StartRL}]')
                else:
                    # 3.1] Spray control
                    if True:
                        pos = self.CMem.PZRSprayPosControl + 0.015 * np.clip(AMod[0] * 2, -2, 2)
                        zip_spray_pos = (['BPRZSP'], [pos])
                        self._send_control_save(zip_spray_pos)
                        a_log_f(s=f'Change Spray Pos [{self.CMem.PZRSprayPosControl:10}|{pos:10}]')
                    # 3.2] Aux Feed
                    if True:
                        aux123 = 0
                        if AMod[1] < -0.3:
                            # Decrease
                            aux123 = -1
                        elif -0.3 <= AMod[1] < 0.3:
                            # Stay
                            aux123 = 0
                        elif 0.3 <= AMod[1]:
                            # Increase
                            aux123 = 1
                        pos1 = self.CMem.Aux1Flow + 0.04 * aux123
                        pos2 = self.CMem.Aux2Flow + 0.04 * aux123
                        pos3 = self.CMem.Aux3Flow + 0.04 * aux123
                        zip_aux_pos = (['WAFWS1', 'WAFWS2', 'WAFWS3'], [pos1, pos2, pos3])
                        self._send_control_save(zip_aux_pos)
                        a_log_f(s=f'AuxFlow'
                                  f'[{self.CMem.Aux1Flow:10}|{pos1:10}]'
                                  f'[{self.CMem.Aux2Flow:10}|{pos2:10}]'
                                  f'[{self.CMem.Aux3Flow:10}|{pos3:10}]')
                    # 3.3] SI Supply water
                    if True:
                        if AMod[2] < -0.8:
                            self._send_control_save(ActOrderBook['CloseSI'])
                            a_log_f(s=f'CloseSI')
                        elif -0.8 <= AMod[2] < -0.6:
                            self._send_control_save(ActOrderBook['StopCHP2'])
                            a_log_f(s=f'StopCHP2')
                        elif -0.6 <= AMod[2] < 0.6:
                            #
                            pass
                        elif 0.6 <= AMod[2] < 0.8:
                            self._send_control_save(ActOrderBook['RunCHP2'])
                            a_log_f(s=f'RunCHP2')
                        elif 0.8 <= AMod[2]:
                            self._send_control_save(ActOrderBook['OpenSI'])
                            a_log_f(s=f'OpenSI')
                    # 3.4] Steam Dump
                    if True:
                        SteamDumpRate = 4
                        DumpPos = self.CMem.PMSS + 2.0E5 * np.clip(AMod[3] * SteamDumpRate,
                                                                   - SteamDumpRate, SteamDumpRate) * 0.2
                        zip_Dump_pos = (['PMSS'], [DumpPos])
                        self._send_control_save(zip_Dump_pos)
                        a_log_f(s=f'PMSS [{self.CMem.PMSS:10}|{DumpPos:10}]')

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
        if self.CMem.CoolingRateSW == 0:
            # 강화학습 이전 시 100 tick
            self.want_tick = int(100)
        else:
            # Cooling 계산 시작 및 강화학습 진입 시 100 tick
            self.want_tick = int(200)

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
        super(ENVCNS, self).reset(initial_nub=1, mal=True, mal_case=12, mal_opt=10005, mal_time=30, file_name=file_name)
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
