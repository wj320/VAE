import sys
import torch
class Logger(object):
    def __init__(self, filename="log/log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w",encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def psnr(im1, im2, bitdepth=8):
    mse = torch.mean((im1.flatten() - im2.flatten())**2)
    return 10 * torch.log10((2**bitdepth-1)/mse)





