from Model_0_Basic.ENVS.CNS_Basic import CNS
from Model_2_Start_up_PZR.ENVS.PID_Na import PID

import numpy as np
import time
import random
import copy


class CMem:
    def __init__(self, mem):
        self.m = mem  # Line CNSmem -> getmem

        self.StartRL = 1

        self.AllRodOut = False
        self.BuxFix_ini = False

        self.update()

    def update(self):
        self.CTIME = self.m['KCNTOMS']['Val']  # CNS Time

        self.Reactor_power = self.m['QPROREL']['Val']   # 0.02 ~ 1.00
        self.TRIP = self.m['KRXTRIP']['Val']
        # 온도 고려 2019-11-04
        self.Tref_Tavg = self.m['ZINST15']['Val']  # Tref-Tavg
        self.Tavg = self.m['UAVLEGM']['Val']  # 308.21
        self.Tref = self.m['UAVLEGS']['Val']  # 308.22
        # 제어봉 Pos
        self.rod_pos = [self.m[nub_rod]['Val'] for nub_rod in ['KBCDO10', 'KBCDO9', 'KBCDO8', 'KBCDO7']]

        if self.AllRodOut == False:
            Sw = True if self.rod_pos[0] == 228 else False
            Sw = True if self.rod_pos[1] == 228 else False
            Sw = True if self.rod_pos[2] == 228 else False
            Sw = True if self.rod_pos[3] >= 216 else False
            self.AllRodOut = Sw

        # self.charging_valve_state = self.m['KLAMPO95']['Val']
        self.main_feed_valve_1_state = self.m['KLAMPO147']['Val']
        self.main_feed_valve_2_state = self.m['KLAMPO148']['Val']
        self.main_feed_valve_3_state = self.m['KLAMPO149']['Val']
        self.main_feed_valves_state = self.m['KLAMPO147']['Val'] + self.m['KLAMPO148']['Val'] + self.m['KLAMPO149']['Val']
        self.vct_level = self.m['ZVCT']['Val']
        # self.pzr_level = self.m['ZINST63']['Val']
        #
        self.Turbine_setpoint = self.m['KBCDO17']['Val']
        self.Turbine_ac = self.m['KBCDO18']['Val']  # Turbine ac condition
        self.Turbine_real = self.m['KBCDO19']['Val']
        self.load_set = self.m['KBCDO20']['Val']  # Turbine load set point
        self.load_rate = self.m['KBCDO21']['Val']  # Turbine load rate
        self.Mwe_power = self.m['KBCDO22']['Val']
        #
        self.Netbreak_condition = self.m['KLAMPO224']['Val']  # 0 : Off, 1 : On
        self.trip_block = self.m['KLAMPO22']['Val']  # Trip block condition 0 : Off, 1 : On
        #
        self.steam_dump_condition = self.m['KLAMPO150']['Val']  # 0: auto 1: man
        self.heat_drain_pump_condition = self.m['KLAMPO244']['Val']  # 0: off, 1: on
        self.main_feed_pump_1 = self.m['KLAMPO241']['Val']  # 0: off, 1: on
        self.main_feed_pump_2 = self.m['KLAMPO242']['Val']  # 0: off, 1: on
        self.main_feed_pump_3 = self.m['KLAMPO243']['Val']  # 0: off, 1: on
        self.cond_pump_1 = self.m['KLAMPO181']['Val']  # 0: off, 1: on
        self.cond_pump_2 = self.m['KLAMPO182']['Val']  # 0: off, 1: on
        self.cond_pump_3 = self.m['KLAMPO183']['Val']  # 0: off, 1: on
        #
        self.ax_off = self.m['CAXOFF']['Val']
        # Boron control
        self.BoronConcen = self.m['KBCDO16']['Val']

        self.BoronManMode = self.m['KLAMPO84']['Val']
        self.BoronBorMode = self.m['KLAMPO84']['Val']
        self.BoronAutMode = self.m['KLAMPO84']['Val']
        self.BoronAILMode = self.m['KLAMPO84']['Val']
        self.BoronDILMode = self.m['KLAMPO84']['Val']

        self.BoronTank = self.m['EBOAC']['Val']
        self.MakeTank  = self.m['EDEWT']['Val']

        self.BoronValve = self.m['WBOAC']['Val']    # Max 2.5
        self.BoronValveOpen = float(np.clip(self.BoronValve + 1, 0, 1))
        self.BoronValveClose = float(np.clip(self.BoronValve - 1, 0, 1))

        self.MakeUpValve = self.m['WDEWT']['Val']   # Max 10
        self.MakeUpValveeOpen = float(np.clip(self.MakeUpValve + 1, 0, 10))
        self.MakeUpValveClose = float(np.clip(self.MakeUpValve - 1, 0, 10))

        # self.CDelt = self.m['TDELTA']['Val']
        # self.PZRPres = self.m['ZINST65']['Val']
        # self.PZRLevl = self.m['ZINST63']['Val']
        # self.PZRTemp = self.m['UPRZ']['Val']
        # self.ExitCoreT = self.m['UUPPPL']['Val']

        self.FV122 = self.m['BFV122']['Val']
        self.FV122M = self.m['KLAMPO95']['Val']

        # self.HV142 = self.m['BHV142']['Val']
        # self.HV142Flow = self.m['WRHRCVC']['Val']
        #
        # self.PZRSprayPos = self.m['ZINST66']['Val']
        #
        # self.LetdownSet = self.m['ZINST36']['Val']  # Letdown setpoint
        # self.LetdownSetM = self.m['KLAMPO89']['Val']  # Letdown setpoint Man0/Auto1
        #
        # self.PZRBackUp = self.m['KLAMPO118']['Val']
        # self.PZRPropo = self.m['QPRZH']['Val']

        # Logic
        if self.CTIME == 0:
            self.AllRodOut = False
            self.BuxFix_ini = False
            self.StartRL = 1

        # if self.AllRodOut == False:
        self.Ref_P = 0.02
        self.Ref_UpP = self.Ref_P + 0.01 # 2% + 1% -> 3%
        self.Ref_UpDis = abs(self.Reactor_power - self.Ref_UpP)   # 0 ~ 0.01
        self.Ref_DoP = self.Ref_P - 0.01 # 2% - 1% -> 1%
        self.Ref_DoDis = abs(self.Reactor_power - self.Ref_DoP)     # 0 ~ 0.01
        self.Out_Ref = True if (self.Ref_UpP - self.Reactor_power) < 0 \
                               or (self.Reactor_power - self.Ref_DoP) < 0 else False


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
            ('QPROREL',     1,      0,      0),       # Reactor power
            ('KBCDO22',     1000,   0,      0),       # MWe power
            ('KBCDO20',     100,    0,      0),       # Load set point
            ('UAVLEGM',     1000,   0,      0),       # T average
            ('KBCDO10',     228,    0,      0),       # Rod Pos 0
            ('KBCDO9',      228,    0,      0),       # Rod Pos 1
            ('KBCDO8',      228,    0,      0),       # Rod Pos 2
            ('KBCDO7',      228,    0,      0),       # Rod Pos 3

            ('DRefPower',       1,      0,      0),         # Reference Power
            ('DRefUpPower',     1,      0,      0),         # Reference Up Power
            ('DRefDownPower',   1,      0,      0),         # Reference Down Power

            # ('DCurrent_t_ref',      1,    0,      0),       #
            # ('DUpDeadBand',         1,    0,      0),       #
            # ('DDownDeadBand',       1,    0,      0),       #
            # ('DUpOperationBand',    1,    0,      0),       #
            # ('DDownOperationBand',  1,    0,      0),       #
        ]

        self.action_space = 1       # Boron
        self.observation_space = len(self.input_info) * self.time_leg
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

    def get_t_ref(self):
        # 평균온도에 기반한 출력 증가 알고리즘
        # (290.2~308.2: 18도 증가) -> ( 2%~100%: 98% 증가 )
        # 18 -> 98 따라서 1%증가시 요구되는 온도 증가량 18/98
        # 1분당 1% 증가시 0.00306 도씩 초당 증가해야함.
        # 2% start_ref_temp = 290.2 매틱 마다 0.00306 씩 증가
        # increase_slop = 0.0001(5배에서 시간당 1%임).
        #               = 0.001 (5배에서 시간당 10%?, 분당 약 0.46~0.5%, 0.085도/분) - Ver10
        #               = 0.001 (5배에서 시간당 10%?, 분당 약 0.46~0.5%, ?도/분) - Ver11

        increase_slop = 0.001489
        start_2per_temp = 291.97
        current_t_ref = start_2per_temp + (increase_slop) * self.CMem.CTIME

        return current_t_ref, current_t_ref + 1, current_t_ref - 1, current_t_ref + 3, current_t_ref - 3

    # ENV RL TOOLs =====================================================================================================
    def get_state(self):
        state = []
        for para, x_round, x_min, x_max in self.input_info:
            if para in self.mem.keys():
                _ = self.mem[para]['Val']
            else:
                ctref, UpD, DownD, UpOp, DownOp = self.get_t_ref()

                # _ = ctref if para == 'DCurrent_t_ref' else None
                # _ = UpD if para == 'DUpDeadBand' else None
                # _ = DownD if para == 'DDownDeadBand' else None
                # _ = UpOp if para == 'DUpOperationBand' else None
                # _ = DownOp if para == 'DDownOperationBand' else None

                if para == 'DRefPower':
                    _ = self.CMem.Ref_P
                elif para == 'DRefUpPower':
                    _ = self.CMem.Ref_UpP
                elif para == 'DRefDownPower':
                    _ = self.CMem.Ref_DoP
                else:
                    _ = None

                # ------------------------------------------------------------------------------------------------------
                if _ is None:
                    raise ValueError(f'{para} is not in self.input_info -> {_}')
                # ------------------------------------------------------------------------------------------------------

            state.append(self.normalize(_, x_round, x_min, x_max))
        return np.array(state), state

    def get_reward(self, A):
        """
        :param A: tanh (-1 ~ 1) 사이 값
        :return:
        """
        r, r_1, r_2, r_3, r_4, r_5 = 0, 0, 0, 0, 0, 0
        # --------------------------------------------------------------------------------------------------------------
        # r_1, 2] 현재 temperature 가 출력 기준 선보다 높아지거나, 낮아지면 거리만큼의 차가 보상으로 제공

        ctref, UpD, DownD, UpOp, DownOp = self.get_t_ref()

        if self.CMem.Tavg > ctref:
            r_1 += (UpOp - self.CMem.Tavg) / 100
        else:
            r_1 += (self.CMem.Tavg - DownOp) / 100

        if DownD <= self.CMem.Tavg <= UpD:
            if self.CMem.Tavg > ctref:
                r_2 += (UpD - self.CMem.Tavg) / 200
            else:
                r_2 += (self.CMem.Tavg - DownD) / 200

        # r_3] 출력 증가율에 따른 보상 선정
        r_3 += min(self.CMem.Ref_UpDis, self.CMem.Ref_DoDis) * 100    # 경계에 가까워 지면 점점 보상이 0으로 감소함. [0, 1]

        r_w = [0 * r_1, 0 * r_2, 1 * r_3, 0 * r_4, 0 * r_5]
        r = sum(r_w)
        # --------------------------------------------------------------------------------------------------------------
        self.Loger_txt += f'R|{r}|{r_1}|{r_2}|{r_3}|{r_4}|{r_5}|'
        return r

    def get_done(self, r, AMod):
        d, ep_d = False, False
        ctref, UpD, DownD, UpOp, DownOp = self.get_t_ref()

        cond = {
            1: True if self.CMem.Tavg < DownOp else False,
            2: True if UpOp < self.CMem.Tavg else False,

            3: self.CMem.Out_Ref,
            4: self.CMem.AllRodOut,
        }
        # --------------------------------------------------------------------------------------------------------------
        # 1] 불만족시 즉각 Done
        if cond[4]:
            # 계속 진행
            if cond[3]:
                d = True
            else:
                ep_d = True
        else:
            if cond[3]:
                d = True
            else:
                pass
        # --------------------------------------------------------------------------------------------------------------
        if d: print(cond)
        # --------------------------------------------------------------------------------------------------------------
        self.Loger_txt += f'D|{d}|{ep_d}|{cond}|'
        return d, ep_d, r

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

            # MainFeedWaterValve
            'MFW_ValveAuto': (['KSWO171', 'KSWO165', 'KSWO159'], [0, 0, 0]),

            # Rod Control ----------------------------------------------------------------------------------------------
            'RodOut':  (['KSWO33', 'KSWO32'], [1, 0]),
            'RodStay': (['KSWO33', 'KSWO32'], [0, 0]),
            'RodIn':   (['KSWO33', 'KSWO32'], [0, 1]),

            # Boron Control --------------------------------------------------------------------------------------------
            # 1] Make-up 주입
            #    1. ALT DIL 로 모드 전환
            #    2. WDEWT로 밸브 개도 전환

            'BoronManMode': (['KSWO74'], [1]),
            'BoronBorMode': (['KSWO75'], [1]),
            'BoronAutMode': (['KSWO76'], [1]),
            'BoronAILMode': (['KSWO77'], [1]),
            'BoronDILMode': (['KSWO78'], [1]),

            'FillBoron': (['EBOAC'], [10000]),
            'FillMakeup': (['EDEWT'], [10000]),

            'BoronValveOpen': (['WBOAC'], [self.CMem.BoronValveOpen]),
            'BoronValveClose': (['WBOAC'], [self.CMem.BoronValveClose]),
            'BoronMakeUpOpen': (['WDEWT'], [self.CMem.MakeUpValveeOpen]),
            'BoronMakeUpClose': (['WDEWT'], [self.CMem.MakeUpValveClose]),

            'BoronValvesReset': (['WBOAC', 'WDEWT'], [0, 0]),

            'BugFix1': (['KSWO86'], [0]),
            # Delta Time -----------------------------------------------------------------------------------------------
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
        AMod = A
        # Action Logger ------------------------------------------------------------------------------------------------
        cr_time = time.strftime('%c', time.localtime(time.time()))
        self.a_log = ''

        def a_log_f(s=''):
            self.a_log += f'[{cr_time}]|[{self.Mother_current_ep:06}]\t|{s}\n'
        # --------------------------------------------------------------------------------------------------------------
        a_log_f(s=f'[Step|{self.ENVStep:10}][{"="*20}]')

        # 0] Bug fix : 17번 조건 초기 조건 재설정
        if self.CMem.BuxFix_ini == False:
            self._send_control_save(ActOrderBook['BugFix1'])            # 보론 밸브 고장 수정
            self._send_control_save(ActOrderBook['FillBoron'])          # 보론 탱크 물 0 -> 10000
            self._send_control_save(ActOrderBook['BoronValvesReset'])   # 보론/Make-up 주입 Valve pos Reset
            a_log_f(s=f'[{"Common":10}] Fix Ini Bugs')
            self.CMem.BuxFix_ini = True
        else:
            # 0] 주급수 및 CVCS 자동으로 전환
            if self.CMem.FV122M == 1:
                self._send_control_save(ActOrderBook['ChargingValveAUto'])
                a_log_f(s=f'[{"Common":10}] ChargingValveAUto [{self.CMem.FV122M}]')
            if self.CMem.main_feed_valves_state != 0:
                self._send_control_save(ActOrderBook['MFW_ValveAuto'])
                a_log_f(s=f'[{"Common":10}] MFW_ValveAuto [{self.CMem.main_feed_valves_state}]')

            # 1] All Rod Out 이전 2% 내 보론 농도 조절 -------------------------------------------------------------------
            if self.CMem.AllRodOut == False:
                """
                목표
                - 일정 간격으로 제어봉 인출
                - 제어봉 인출에 따른 출력 증가를 보론 주입을 통해 감쇄
                """
                # 1-1] 일정 간격으로 제어봉 인출
                if self.ENVStep % 10 == 0: # 매 3 ENVStep * tick 마다 제어봉 증가
                    self._send_control_save(ActOrderBook['RodOut'])
                    a_log_f(s=f'[{"NoRodOut":10}] '
                              f'Rod Out ['
                              f'{self.CMem.rod_pos[0]:4}|'
                              f'{self.CMem.rod_pos[1]:4}|'
                              f'{self.CMem.rod_pos[2]:4}|'
                              f'{self.CMem.rod_pos[3]:4}]')
                # 1-2] 제어봉 인출에 따른 출력 증가를 보론 주입을 통해 감쇄
                # Boron Valve ------------------------------------------------------------------------------------------
                if AMod[0] < 0:
                    # Inject
                    self._send_control_save(ActOrderBook['BoronValveOpen'])
                    a_log_f(s=f'[{"NoRodOut":10}] BoronValveOpen')
                else:
                    # Stay
                    a_log_f(s=f'[{"NoRodOut":10}] BoronValveStay')
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

        self.want_tick = int(295)

        # New Data (time t+1) -------------------------------------
        super(ENVCNS, self).step()                  # 전체 CNS mem run-Freeze 하고 mem 업데이트
        self.CMem.update()                          # 선택 변수 mem 업데이트

        self._append_val_to_list()
        self.ENVStep += 1

        reward = self.get_reward(AMod)                                      # [r(t+1)]
        done, ep_done, reward = self.get_done(reward, AMod)                 # [d(t+1)]
        next_state, next_state_list = self.get_state()                      # [s(t+1)]
        self.Loger_txt += f'NS|{next_state_list}|'                          #
        # ----------------------------------------------------------
        self.ENVlogging(s=self.Loger_txt)
        self.state_txt = next_state_list                                    # [s(t) <- s(t+1)]
        self.Loger_txt = ''

        # 벨브 Pos 초기화
        self.want_tick = int(5)
        super(ENVCNS, self).step()  # 전체 CNS mem run-Freeze 하고 mem 업데이트
        self.CMem.update()  # 선택 변수 mem 업데이트

        self._append_val_to_list()

        self._send_control_save((['WBOAC', 'WDEWT'], [0, 0]))
        self._send_control_to_cns()
        return next_state, reward, done, ep_done, AMod

    def reset(self, file_name, current_ep):
        # 1] CNS 상태 초기화 및 초기화된 정보 메모리에 업데이트
        super(ENVCNS, self).reset(initial_nub=17, mal=False, mal_case=0, mal_opt=0, mal_time=0, file_name=file_name)
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
