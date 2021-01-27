import socket


def get_cns_info():
    """
    현재 컴퓨터의 ip를 확인후 이에 따라 CNS 환경 정보 반환
    :return:
    """
    com_ip = socket.gethostbyname(socket.getfqdn())

    if com_ip == '192.168.0.29':
        return {
            0: {'CNSIP': '192.168.0.101', 'COMIP': '192.168.0.29', 'PORT': 7101, 'PID': False},
            1: {'CNSIP': '192.168.0.101', 'COMIP': '192.168.0.29', 'PORT': 7102, 'PID': False},
            2: {'CNSIP': '192.168.0.101', 'COMIP': '192.168.0.29', 'PORT': 7103, 'PID': False},
            3: {'CNSIP': '192.168.0.101', 'COMIP': '192.168.0.29', 'PORT': 7104, 'PID': False},
        }
    else:
        return {
            0: {'CNSIP': '192.168.0.211', 'COMIP': '192.168.0.200', 'PORT': 7101, 'PID': False},
            1: {'CNSIP': '192.168.0.211', 'COMIP': '192.168.0.200', 'PORT': 7102, 'PID': False},
            2: {'CNSIP': '192.168.0.211', 'COMIP': '192.168.0.200', 'PORT': 7103, 'PID': False},
            3: {'CNSIP': '192.168.0.211', 'COMIP': '192.168.0.200', 'PORT': 7104, 'PID': False},
            4: {'CNSIP': '192.168.0.211', 'COMIP': '192.168.0.200', 'PORT': 7105, 'PID': False},

            5: {'CNSIP': '192.168.0.212', 'COMIP': '192.168.0.200', 'PORT': 7201, 'PID': False},
            6: {'CNSIP': '192.168.0.212', 'COMIP': '192.168.0.200', 'PORT': 7202, 'PID': False},
            7: {'CNSIP': '192.168.0.212', 'COMIP': '192.168.0.200', 'PORT': 7203, 'PID': False},
            8: {'CNSIP': '192.168.0.212', 'COMIP': '192.168.0.200', 'PORT': 7204, 'PID': False},
            9: {'CNSIP': '192.168.0.212', 'COMIP': '192.168.0.200', 'PORT': 7205, 'PID': False},

            10: {'CNSIP': '192.168.0.213', 'COMIP': '192.168.0.200', 'PORT': 7301, 'PID': False},
            11: {'CNSIP': '192.168.0.213', 'COMIP': '192.168.0.200', 'PORT': 7302, 'PID': False},
            12: {'CNSIP': '192.168.0.213', 'COMIP': '192.168.0.200', 'PORT': 7303, 'PID': False},
            13: {'CNSIP': '192.168.0.213', 'COMIP': '192.168.0.200', 'PORT': 7304, 'PID': False},
            14: {'CNSIP': '192.168.0.213', 'COMIP': '192.168.0.200', 'PORT': 7305, 'PID': False},
        }


if __name__ == '__main__':
    print(get_cns_info())
