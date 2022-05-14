from torch import nn


class MoveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = 1


class DQN:
    def __init__(self, net):
        self.eval_net, self.target_net = net(), net()

    def choose_action(self, x):
        pass
