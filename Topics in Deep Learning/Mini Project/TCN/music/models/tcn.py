from torch import nn
from TCN.residual_block import ResidualBlock

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = ResidualBlock(input_size, hidden_size, kernel_size, dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).double()
        return self.sig(output)
