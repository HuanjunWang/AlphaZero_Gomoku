import time
from multiprocessing import Process
import numpy as np



class SelfPlayer(Process):
    def __init__(self):
        super(SelfPlayer, self).__init__()
        self.num = 0

    def run(self):
        self.num = np.random.randn()
        print("In process ", self.num)
        time.sleep(1)




if __name__ == "__main__":

    selfplayers = [SelfPlayer() for _ in range(4)]

    for p in selfplayers:
        p.start()


    for p in selfplayers:
        print(p.num)


    for p in selfplayers:
        p.join()




