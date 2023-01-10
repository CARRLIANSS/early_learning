from loss_funcs import *

class DeepDA_Losses(nn.Module):

    def __init__(self, loss_type, **kwargs):
        super(DeepDA_Losses, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mmd":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "coral":
            self.loss_func = CORAL
        elif loss_type == "bnm":
            self.loss_func = BNM
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: 0  # return 0

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)