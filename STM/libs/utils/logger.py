import time
import logging


def getLogger(name=None):
    if name is not None:
        return logging.getLogger(name)
    else:
        return logging.getLogger()

def setup(filename, resume=False):

    log_format = '[%(levelname)s][%(asctime)s][%(name)s:%(lineno)d] %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        style='%'
        )

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, mode='a' if resume else 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(fmt=log_format)
        )
    rootLogger.addHandler(fh)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count