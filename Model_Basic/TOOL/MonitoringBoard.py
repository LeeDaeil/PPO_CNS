import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ======================================================================================================================
# Training Board


class TrainingBoard:
    def __init__(self, replay_buffer):
        app = QApplication(sys.argv)
        window = Board(replay_buffer)
        window.show()
        app.exec_()


class Board(QWidget):
    def __init__(self, replay_buffer):
        super().__init__()
        self.replay_buffer = replay_buffer

        self.initUI()

        self.setLayout(self.layout)
        self.setGeometry(200, 200, 800, 600)

        timer = QTimer(self)
        timer.setInterval(2000)
        for _ in [self.update_plot]:
            timer.timeout.connect(_)
        timer.start()

    def initUI(self):
        self.fig = plt.Figure()
        gs = GridSpec(2, 2, figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.layout = layout

    def update_plot(self):
        c1_loss, c2_loss, p_loss, ent_loss, alpha = self.replay_buffer.get_train_info()

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        self.ax1.plot(c1_loss, label='C1_loss')
        self.ax1.plot(c2_loss, label='C2_loss')

        self.ax2.plot(p_loss, label='p_loss')

        self.ax3.plot(ent_loss, label='ent_loss')

        self.ax4.plot(ent_loss, label='ent_loss')

        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()
        self.ax4.legend()

        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

# ======================================================================================================================

