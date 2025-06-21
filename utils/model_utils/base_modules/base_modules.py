import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MLP(nn.Module):
	'''
	base mlp class for use by other modules. uses gelu
	'''

	def __init__(self, d_in=512, d_out=512, d_hidden=1024, hidden_layers=0, dropout=0.1, act="gelu", zeros=False):
		super(MLP, self).__init__()

		self.in_proj = nn.Linear(d_in, d_hidden)
		self.hidden_proj = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for layer in range(hidden_layers)])
		self.out_proj = nn.Linear(d_hidden, d_out)

		self.in_dropout = nn.Dropout(dropout)
		self.hidden_dropout = nn.ModuleList([nn.Dropout(dropout) for layer in range(hidden_layers)])

		if act == "gelu":
			self.act = F.gelu 
		elif act == "silu":
			self.act = F.silu
		elif act == "relu":
			self.act = F.relu
		elif act == "sigmoid":
			self.act = F.sigmoid
		else:
			self.act = lambda x: x # no activation if none of the above 

		self.init_linears(zeros=zeros)

	def init_linears(self, zeros=False):

		init_xavier(self.in_proj)  # Xavier for the first layer

		for layer in self.hidden_proj:
			init_kaiming(layer)  # Kaiming for hidden layers

		if zeros:
			init_zeros(self.out_proj) 
		else:
			init_xavier(self.out_proj)  # Xavier for output layer

	def forward(self, x):
		x = self.in_dropout(self.act(self.in_proj(x)))
		for hidden, dropout in zip(self.hidden_proj, self.hidden_dropout):
			x = dropout(self.act(hidden(x)))
		x = self.out_proj(x) # no activation or dropout on output

		return x


class StaticLayerNorm(nn.Module):
	'''just normalizes each token to have a mean of 0 and var of 1, no scaling and shifting'''
	def __init__(self, d_model):
		super(StaticLayerNorm, self).__init__()
		self.d_model = d_model
	def forward(self, x):
		centered = x - x.mean(dim=2, keepdim=True) 
		std = centered.std(dim=2, keepdim=True)
		std = std.masked_fill(std==0, 1)
		return centered / std



# initializations for linear layers
def init_orthogonal(m):
	if isinstance(m, nn.Linear):
		init.orthogonal_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_kaiming(m):
	if isinstance(m, nn.Linear):
		init.kaiming_uniform_(m.weight, nonlinearity='relu')
		if m.bias is not None:
			init.zeros_(m.bias)
def init_xavier(m):
	if isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_zeros(m):
	if isinstance(m, nn.Linear):
		init.zeros_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
