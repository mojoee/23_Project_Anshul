import torch.nn as nn
from torch.autograd import Variable
import torch


class LSTMRegressor(nn.Module):
	def __init__(self, batch_size, input_size, output_size, hidden_size):
		super(LSTMRegressor, self).__init__()
		self.batch_size = batch_size
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		
		self.lstm = nn.LSTM(input_size, hidden_size) # Our main hero for this tutorial
		self.label = nn.Linear(hidden_size, output_size)
		
	def forward(self, input, batch_size=None):
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) 
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) 
		else:
			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		output, (final_hidden_state, final_cell_state) = self.lstm(input.view(1, len(input), -1), (h_0, c_0))
		final_output = self.label(final_hidden_state[-1]) 
		
		return final_output