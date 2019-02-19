import time

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPainter

import sys

from threading import Thread


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


class MyWidget(QtWidgets.QWidget):
    def __init__(self, size=8):
        super().__init__()
        self.size = size
        self.width = 600
        self.step = self.width // (self.size - 1)
        self.margin = self.step
        self.states = []
        self.setGeometry(300, 300, self.width + self.margin * 2, self.width + self.margin * 2)
        self.show()



    def drawBoard(self):
        self.qp = QtGui.QPainter(self)
        br = QtGui.QBrush(QtGui.QColor(100, 10, 10, 40))
        self.qp.setBrush(br)

        begin = QtCore.QPoint(self.margin, self.margin)
        end = QtCore.QPoint(self.margin + self.width, self.margin + self.width)

        self.qp.drawRect(QtCore.QRect(begin, end))
        for i in range(1, self.size - 1):
            self.qp.drawLine(self.margin, self.margin + self.step * i, self.margin + self.width, self.margin + self.step * i)
            self.qp.drawLine(self.margin + self.step * i, self.margin, self.margin + self.step * i, self.margin + self.width)

        for p in self.states:
            if p['player'] == 1:
                self.qp.setBrush(QtCore.Qt.black)
            else:
                self.qp.setBrush(QtCore.Qt.white)

                x = self.margin + self.step * p['x']
                y = self.margin + self.step * p['y']
                self.qp.drawEllipse(QtCore.QPoint(x, y), self.step / 3, self.step / 3)

    def add_point(self, point):
        print("add point", point)
        self.states.append(point)
        #self.update()


    def paintEvent(self, event):
        print("paintEvent")
        self.drawBoard()


    def mousePressEvent(self, event):
        self.update()

    def mouseMoveEvent(self, event):
        self.update()

    def mouseReleaseEvent(self, event):
        self.update()


class Game(object):
    def __init__(self):
        gui_thread = Thread(target=self.gui_thread)
        gui_thread.daemon = True
        gui_thread.start()

    def gui_thread(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = MyWidget()
        self.window.show()

        self.app.exec_()

    def step(self):
        time.sleep(1)
        p = {'player': 1, 'x': 5, 'y': 5}
        self.window.add_point(p)
        time.sleep(10)


if __name__ == '__main__':
    import sys
    sys.excepthook = except_hook
    game = Game()
    game.step()
