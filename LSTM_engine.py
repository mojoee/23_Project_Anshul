import torch.nn as nn
from torch.autograd import Variable


class MyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        self.input_size=input_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)


    def forward(self):


        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())


        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))


        return self.label(final_hidden_state[-1]) 
