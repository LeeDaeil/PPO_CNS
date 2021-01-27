import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from Model_1_Emergency.ENVS.PTCurve import PTCureve

# ======================================================================================================================
# Training Board


class ParaBoard:
    def __init__(self, replay_buffer, nub_agent):
        app = QApplication(sys.argv)
        window = Board(replay_buffer, nub_agent)
        window.show()
        app.exec_()

class BoardUI(QWidget):
    def __init__(self, nub_agent=30, nub_gp=5):
        super(BoardUI, self).__init__()
        self.setGeometry(200, 200, 800, 600)
        self.nub_agent = nub_agent
        self.nub_gp = nub_gp
        self.selected_nub = 0

        self.abs_path = 'C:/Users/Com/Desktop/DL_Code/Soft_Actor_Critic_Agent_CNS/DB/'

        self.initUI()

    def initUI(self):
        #
        # self.fig = plt.Figure()
        # self.axs = []
        #
        # required_col = 2
        # required_row = 3
        #
        # required_row = (self.nub_gp // required_col) + 1
        # gs = GridSpec(required_row, required_col, figure=self.fig)
        #
        # count_gp_nub = 0
        # for r in range(required_row):
        #     for i in range(required_col):
        #         if count_gp_nub < self.nub_gp:
        #             self.axs.append(self.fig.add_subplot(gs[r, i]))
        #             count_gp_nub += 1

        self.fig = plt.Figure()
        required_row = 5
        required_col = 2
        gs = GridSpec(required_row, required_col, figure=self.fig)

        self.axs = [
            self.fig.add_subplot(gs[0, 0]),  # 에이전트 누적 Reward
            self.fig.add_subplot(gs[0, 1]),  # 현재 보상
            self.fig.add_subplot(gs[1, 0]),  # 현재 수위
            self.fig.add_subplot(gs[1, 1]),  # 현재 급수량
            self.fig.add_subplot(gs[2:5, :], projection='3d'),
        ]

        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()
        self.canvas = FigureCanvas(self.fig)
        # --------------------------------------------------------------------------------------------------------------
        # Layout set-up
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left button
        self.button_area_wid_lay = QVBoxLayout()
        self.button_area_wid_lay.setContentsMargins(0, 0, 0, 0)

        self.line_ed = QLineEdit('FigName')
        self.button_area_wid_lay.addWidget(self.line_ed)

        b = QPushButton(f'SaveFIG')
        b.clicked.connect(self.click_save_fig)
        self.button_area_wid_lay.addWidget(b)

        for i in range(self.nub_agent):
            b = QPushButton(f'{i}')
            b.clicked.connect(self.click_button)
            self.button_area_wid_lay.addWidget(b)

        self.button_area_wid = QWidget()
        self.button_area_wid.setFixedWidth(150)
        self.button_area_wid.setLayout(self.button_area_wid_lay)

        button_width = self.button_area_wid.width() + 5

        # Scroll area
        scroll_area_wid_gp = QVBoxLayout()
        scroll_area_wid_gp.setContentsMargins(0, 0, 0, 0)
        scroll_area_wid_gp.addWidget(self.canvas)

        self.scroll_area_wid = QWidget()
        self.scroll_area_wid.setLayout(scroll_area_wid_gp)
        self.scroll_area_wid.setFixedSize(self.width() - button_width, required_row * 300)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_wid)

        # main layout
        main_layout.addWidget(self.button_area_wid)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

    def click_button(self):
        self.selected_nub = self.sender().text()

    def click_save_fig(self):
        if self.line_ed.text() != '':
            self.fig.savefig(f'{self.abs_path}/TRAIN_INFO/{self.line_ed.text()}.svg', format='svg', dpi=1200)
        else:
            self.fig.savefig(f'{self.abs_path}/TRAIN_INFO/SaveFIg.svg', format='svg', dpi=1200)

    def resizeEvent(self, e):
        button_width = self.button_area_wid.width() + 5
        self.scroll_area_wid.setFixedWidth(self.width() - 20 - button_width)


class Board(BoardUI):
    def __init__(self, replay_buffer, nub_agent):
        super(Board, self).__init__(nub_agent=nub_agent)

        self.replay_buffer = replay_buffer

        timer = QTimer(self)
        timer.setInterval(2000)
        for _ in [self.update_plot]:
            timer.timeout.connect(_)
        timer.start()

    def update_plot(self):
        try:
            KCNTOMS, UAVLEG2, ZINST65, WFWLN12, CoolingRateSW, Reward = self.replay_buffer.get_para(f'{self.selected_nub}')
            _ = [ax.clear() for ax in self.axs]

            if len(KCNTOMS) > 1:
                self.axs[0].plot(Reward)  # Reward
                self.axs[0].grid()

                self.axs[1].plot(WFWLN12)  # Reward
                self.axs[1].grid()

                # 인디 케이터

                inv_KCNTOMS = [- val for val in KCNTOMS]

                Temp = []
                UpPres = []
                BotPres = []
                for _ in range(0, 350):
                    uppres, botpres = PTCureve()._get_pres(_)
                    Temp.append([_])
                    UpPres.append([uppres])
                    BotPres.append([botpres])

                PTX = np.array(Temp)
                BotZ = np.array(BotPres)
                UpZ = np.array(UpPres)
                PTY = np.array([[0] for _ in range(0, 350)])

                PTX = np.hstack([PTX[:, 0:1], Temp])
                BotZ = np.hstack([BotZ[:, 0:1], BotPres])
                UpZ = np.hstack([UpZ[:, 0:1], UpPres])
                PTY = np.hstack([PTY[:, 0:1], np.array([[inv_KCNTOMS[-1]] for _ in range(0, 350)])])

                zero = [0 for _ in range(len(UAVLEG2))]

                self.axs[4].plot3D(CoolingRateSW, inv_KCNTOMS, zero, color='orange', lw=1.5, ls='--')

                self.axs[4].plot3D([170, 0, 0, 170, 170],
                                   [inv_KCNTOMS[-1], inv_KCNTOMS[-1], 0, 0, inv_KCNTOMS[-1]],
                                   [29.5, 29.5, 29.5, 29.5, 29.5], color='black', lw=0.5, ls='--')
                self.axs[4].plot3D([170, 0, 0, 170, 170],
                                   [inv_KCNTOMS[-1], inv_KCNTOMS[-1], 0, 0, inv_KCNTOMS[-1]],
                                   [17, 17, 17, 17, 17], color='black', lw=0.5, ls='--')
                self.axs[4].plot3D([170, 170], [inv_KCNTOMS[-1], inv_KCNTOMS[-1]],
                                   [17, 29.5], color='black', lw=0.5, ls='--')
                self.axs[4].plot3D([170, 170], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')
                self.axs[4].plot3D([0, 0], [inv_KCNTOMS[-1], inv_KCNTOMS[-1]], [17, 29.5], color='black', lw=0.5, ls='--')
                self.axs[4].plot3D([0, 0], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')

                self.axs[4].plot_surface(PTX, PTY, UpZ, rstride=8, cstride=8, alpha=0.15, color='r')
                self.axs[4].plot_surface(PTX, PTY, BotZ, rstride=8, cstride=8, alpha=0.15, color='r')

                #
                # 3D plot
                self.axs[4].plot3D(UAVLEG2, inv_KCNTOMS, ZINST65, color='blue', lw=1.5)

                # linewidth or lw: float
                self.axs[4].plot3D([UAVLEG2[-1], UAVLEG2[-1]],
                                   [inv_KCNTOMS[-1], inv_KCNTOMS[-1]],
                                   [0, ZINST65[-1]], color='blue', lw=0.5, ls='--')
                self.axs[4].plot3D([0, UAVLEG2[-1]],
                                   [inv_KCNTOMS[-1], inv_KCNTOMS[-1]],
                                   [ZINST65[-1], ZINST65[-1]], color='blue', lw=0.5, ls='--')
                self.axs[4].plot3D([UAVLEG2[-1], UAVLEG2[-1]],
                                   [0, inv_KCNTOMS[-1]],
                                   [ZINST65[-1], ZINST65[-1]], color='blue', lw=0.5, ls='--')
                # each
                self.axs[4].plot3D(UAVLEG2, inv_KCNTOMS, zero, color='black', lw=1, ls='--')  # temp
                self.axs[4].plot3D(zero, inv_KCNTOMS, ZINST65, color='black', lw=1, ls='--')  # pres
                self.axs[4].plot3D(UAVLEG2, zero, ZINST65, color='black', lw=1, ls='--')  # PT

                # 절대값 처리
                self.axs[4].set_yticklabels([int(_) for _ in abs(self.axs[4].get_yticks())])

                self.axs[4].set_xlabel('Temperature')
                self.axs[4].set_ylabel('Time [Tick]')
                self.axs[4].set_zlabel('Pressure')

                self.axs[4].set_xlim(0, 350)
                self.axs[4].set_zlim(0, 200)

            # _ = [ax.legend() for ax in self.axs]
            self.fig.set_tight_layout(True)
            self.fig.canvas.draw()
        except Exception as e:
            print(e)

# ======================================================================================================================
if __name__ == '__main__':
    # Board_Tester
    app = QApplication(sys.argv)
    window = BoardUI()
    window.show()
    app.exec_()
