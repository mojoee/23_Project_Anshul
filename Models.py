import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


class DNN(nn.Module):
    #hl1 100 acti: relu
    #hl2 150 acti: relu
    #hl3 120 acti: relu
    #hl4 120 acti: relu
    def __init__(self,input_size):
      super(DNN, self).__init__()
      self.fc1 = nn.Linear(input_size, 100)
      self.fc2 = nn.Linear(100, 150)
      self.fc3 = nn.Linear(150, 120)
      self.fc4 = nn.Linear(120, 1)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x=x.squeeze()
      x = self.fc1(x)
      x = F.relu(x)

      x = self.fc2(x)
      x = F.relu(x)

      x = self.fc3(x)
      x = F.relu(x)
      
      x = self.fc4(x)
      output = x

      return output