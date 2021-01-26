from Model_Basic.AGENT.Utils import hard_update
from Model_Basic.TOOL.MonitoringBoard import TrainingBoard
from Model_Basic.TOOL.ShAdam import SharedAdam
from FolderManger import Folder_Manager

from Model_Emergency.TOOL.Replay import ReplayMemory
from Model_Emergency.AGENT.SAC_m import SAC as SAC_master
from Model_Emergency.AGENT.SAC_s import SAC as SAC_slave
from Model_Emergency.AGENT.Networks import QNetwork, GaussianPolicy, DeterministicPolicy

from Model_Emergency.ENVS.CNS_Env_Em import ENVCNS
from Model_Emergency.TOOL.PARAMonitoringBoard import ParaBoard

from multiprocessing.managers import BaseManager
from multiprocessing import Process

if __name__ == '__main__':
    Folder_Manager()

    # CNS_info ---------------------------------------------------------------------------------------------------------
    CNS_info = {
        0: {'CNSIP': '192.168.0.101', 'COMIP': '192.168.0.29', 'PORT': 7101, 'PID': False},
        1: {'CNSIP': '192.168.0.101', 'COMIP': '192.168.0.29', 'PORT': 7102, 'PID': False},
    }

    CNS_Envs = [
       ENVCNS(Name=f'Agent_{i}',
              CNS_IP=CNS_info[i]['CNSIP'], CNS_PORT=CNS_info[i]['PORT'],
              Remote_IP=CNS_info[i]['COMIP'], Remote_PORT=CNS_info[i]['PORT']) for i in range(len(CNS_info))
    ]

    print(f'[{"CNS Info":20}][Nub_env|{len(CNS_info):5}]')
    [print(f'[{"CNS Info_IP/PORT":20}][Info|{i:3}_{CNS_info[i]}]') for i in range(len(CNS_info))]
    # Initial Set-up ---------------------------------------------------------------------------------------------------
    lr = 0.0003
    policy_type = "Gaussian"
    automatic_entropy_tuning = False
    print(f'[{"Initial Set-up":20}]'
          f'[lr|{lr:10}]'
          f'[policy_type|{policy_type:10}]'
          f'[automatic_entropy_tuning|{automatic_entropy_tuning:5}]')

    # Shared Network ---------------------------------------------------------------------------------------------------
    sh_net_sw = True
    print(f'[{"Share Network":20}][Net_sh|{sh_net_sw:5}]')
    if sh_net_sw:
        """
        sh_net 는 모든 process 의 에이전트가 하나의 네트워크를 공유한다. 마치 하나의 네트워크가 여러개 환경과 상호작용하는 것임.
        """
        sh_critic = QNetwork(CNS_Envs[0].observation_space,
                             CNS_Envs[0].action_space,
                             hidden_dim=256)
        sh_critic_opt = SharedAdam(sh_critic.parameters(), lr=lr)
        sh_target = QNetwork(CNS_Envs[0].observation_space,
                             CNS_Envs[0].action_space,
                             hidden_dim=256)
        hard_update(sh_target, sh_critic)

        if policy_type == "Gaussian":
            sh_policy = GaussianPolicy(CNS_Envs[0].observation_space,
                                       CNS_Envs[0].action_space,
                                       hidden_dim=256)
            sh_policy_opt = SharedAdam(sh_policy.parameters(), lr=lr)

        else:
            sh_policy = DeterministicPolicy(CNS_Envs[0].observation_space,
                                            CNS_Envs[0].action_space,
                                            hidden_dim=256)
            sh_policy_opt = SharedAdam(sh_policy.parameters(), lr=lr)

        sh_critic.share_memory()
        sh_critic_opt.share_memory()
        sh_target.share_memory()

        sh_policy.share_memory()

        sh_net = {
            'critic': sh_critic,
            'target': sh_target,
            'critic_opt': sh_critic_opt,

            'policy': sh_policy,
            'policy_opt': sh_policy_opt,
        }
    else:
        sh_net = None

    # Replay buffer ----------------------------------------------------------------------------------------------------
    BaseManager.register('ReplayMemory', ReplayMemory)
    manager = BaseManager()
    manager.start()

    replay_buffer = manager.ReplayMemory(capacity=1e6, nub_env=len(CNS_info))
    print(f'[{"Manger Info":20}][ReplayBuffer|{replay_buffer}]')
    # Build Process ----------------------------------------------------------------------------------------------------
    p_list = []
    print(f'[{"SAC_master":20}][Build!!]')
    p = Process(
        target=SAC_master,
        args=(0, CNS_Envs[0], replay_buffer, lr, policy_type, automatic_entropy_tuning, sh_net,),
        daemon=True
    )
    p_list.append(p)

    for i in range(0, len(CNS_info)):
        print(f'[{"SAC_slave"+f"{i}":20}][Build!!]')
        p = Process(
            target=SAC_slave,
            args=(i, CNS_Envs[i], replay_buffer, lr, policy_type, automatic_entropy_tuning, sh_net, ),
            daemon=True
        )
        p_list.append(p)

    p_board = Process(target=TrainingBoard, args=(replay_buffer, ), daemon=True)
    p_list.append(p_board)
    # ------------------------------------------------------------------------------------------------------------------
    p_board = Process(target=ParaBoard, args=(replay_buffer,), daemon=True)         # ParaBoard
    p_list.append(p_board)
    # ------------------------------------------------------------------------------------------------------------------
    [p_.start() for p_ in p_list]
    [p_.join() for p_ in p_list]  # finished at the same time
    # End --------------------------------------------------------------------------------------------------------------
