import torch
from torch import nn

# Experimental stuff that wasn't used in the final solution.

class GridLSTM(nn.Module):
    # Kind of like a regular LSTM except it takes input from 2 dimensions rather than 1 (the cell above it and the cell to its left).
    # Was going to traverse the feature map with this.

    def __init__(self, input_size, hidden_size):
        # Input size 512 for resnet
        super(GridLSTM, self).__init__()
        self.grid_cell = nn.LSTMCell(input_size, hidden_size*2)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, input):
        # Input is the final feature maps (batch_size, features, height, width)
        # converted into (height, width, batch_size, features)
        h0 = torch.autograd.Variable(input.data.new(input.size(2), self.hidden_size).zero_(), requires_grad=False)
        hx_cx = [(h0, h0)] * input.size(1)

        for i, row in enumerate(input):
            h, c = h0, h0
            for j, column in enumerate(row):
                hx, cx = hx_cx[j]
                h2, c2 = self.grid_cell(column, (torch.cat((h, hx), 1), torch.cat((c, cx), 1)))
                h, hx = torch.chunk(h2, 2, dim=1)
                c, cx = torch.chunk(c2, 2, dim=1)
                hx_cx[j] = (hx, cx)

        return h2, c2

class SkipLSTM(nn.Module):
    # LSTM with skip connections to future timesteps. Didn't finish this because I think
    # an attention layer is enough.

    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional):
        super(SkipLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.skip_lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, input):
        hc = None
        for step in input:
            step = torch.unsqueeze(step, 0)
            if not hc:
                output, hc = self.skip_lstm(step)
            else:
                output, hc = self.skip_lstm(step, hc)
        
        return output, hc