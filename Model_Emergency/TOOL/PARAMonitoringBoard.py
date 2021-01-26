import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ======================================================================================================================
# Training Board


class ParaBoard:
    def __init__(self, replay_buffer):
        app = QApplication(sys.argv)
        window = Board(replay_buffer)
        window.show()
        app.exec_()

class BoardUI(QWidget):
    def __init__(self, nub_agent=30, nub_gp=5):
        super(BoardUI, self).__init__()
        self.setGeometry(200, 200, 800, 600)
        self.nub_agent = nub_agent
        self.nub_gp = nub_gp
        self.selected_nub = 0

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
        for i in range(self.nub_agent):
            b = QPushButton(f'{i}')
            b.clicked.connect(self.click_button)
            self.button_area_wid_lay.addWidget(b)

        self.button_area_wid = QWidget()
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

    def resizeEvent(self, e):
        button_width = self.button_area_wid.width() + 5
        self.scroll_area_wid.setFixedWidth(self.width() - 20 - button_width)


class Board(BoardUI):
    def __init__(self, replay_buffer):
        super(Board, self).__init__(nub_agent=2)

        self.replay_buffer = replay_buffer

        timer = QTimer(self)
        timer.setInterval(2000)
        for _ in [self.update_plot]:
            timer.timeout.connect(_)
        timer.start()

    def update_plot(self):
        KCNTOMS, UAVLEG2, ZINST65, WFWLN123 = self.replay_buffer.get_para(f'{self.selected_nub}')
        _ = [ax.clear() for ax in self.axs]

        if len(KCNTOMS) > 1:
            # 인디 케이터
            self.axs[4].plot3D([170, 0, 0, 170, 170],
                               [KCNTOMS[-1], KCNTOMS[-1], 0, 0, KCNTOMS[-1]],
                               [29.5, 29.5, 29.5, 29.5, 29.5], color='black', lw=0.5, ls='--')
            self.axs[4].plot3D([170, 0, 0, 170, 170],
                               [KCNTOMS[-1], KCNTOMS[-1], 0, 0, KCNTOMS[-1]],
                               [17, 17, 17, 17, 17], color='black', lw=0.5, ls='--')
            self.axs[4].plot3D([170, 170], [KCNTOMS[-1], KCNTOMS[-1]],
                               [17, 29.5], color='black', lw=0.5, ls='--')
            self.axs[4].plot3D([170, 170], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')
            self.axs[4].plot3D([0, 0], [KCNTOMS[-1], KCNTOMS[-1]], [17, 29.5], color='black', lw=0.5, ls='--')
            self.axs[4].plot3D([0, 0], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')
            #

            # 3D plot
            self.axs[4].plot3D(UAVLEG2, KCNTOMS, ZINST65, color='blue', lw=1.5)

            # linewidth or lw: float
            self.axs[4].plot3D([UAVLEG2[-1], UAVLEG2[-1]],
                               [KCNTOMS[-1], KCNTOMS[-1]],
                               [0, ZINST65[-1]], color='blue', lw=0.5, ls='--')
            self.axs[4].plot3D([0, UAVLEG2[-1]],
                               [KCNTOMS[-1], KCNTOMS[-1]],
                               [ZINST65[-1], ZINST65[-1]], color='blue', lw=0.5, ls='--')
            self.axs[4].plot3D([UAVLEG2[-1], UAVLEG2[-1]],
                               [0, KCNTOMS[-1]],
                               [ZINST65[-1], ZINST65[-1]], color='blue', lw=0.5, ls='--')

        # _ = [ax.legend() for ax in self.axs]
        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

# ======================================================================================================================
if __name__ == '__main__':
    # Board_Tester
    app = QApplication(sys.argv)
    window = BoardUI()
    window.show()
    app.exec_()
