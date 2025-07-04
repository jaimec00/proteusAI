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

class MPNN(nn.Module):
	def __init__(self, d_model=128, dropout=0.00, use_adaLN=False):
		super(MPNN, self).__init__()

		self.node_messenger = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
		self.node_messenger_norm = nn.LayerNorm(d_model) if not use_adaLN else adaLN(d_model)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_model*4, hidden_layers=0, dropout=dropout, act="gelu", zeros=False)
		self.ffn_norm = nn.LayerNorm(d_model) if not use_adaLN else adaLN(d_model)

		self.edge_messenger = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
		self.edge_messenger_norm = nn.LayerNorm(d_model) if not use_adaLN else adaLN(d_model)

		self.dropout = nn.Dropout(dropout)

	def forward(self, V, E, K, edge_mask=None, t=None):
		'''
		for now copying pmpnn, except node features are initialized with wf embedding instead of zeros. 
		will then try attention w/ receiving nodes as Q, edges as K, and sending nodes as V for the node messenger
		edge updates will be the same as in pmpnn, since dont need attention since only depends on the nodes
		'''

		# prepare the node message
		Mv_pre = self.prepare_message(V, E, K) # Z x N x K x d_model

		# process the node message
		Mv = torch.sum(self.node_messenger(Mv_pre) * edge_mask.unsqueeze(3), dim=2) # Z x N x K x d_model

		if t is None:

			# send the node message
			V = self.node_messenger_norm(V + self.dropout(Mv)) # Z x N x d_model
			
			# process the updated node
			V = self.ffn_norm(V + self.dropout(self.ffn(V)))

		else:

			# send the node message with timestep conditioning
			V = self.node_messenger_norm(V + self.dropout(Mv), t) # Z x N x d_model

			# process the updated node with timestep conditioning
			V = self.ffn_norm(V + self.dropout(self.ffn(V)), t)


		# prepare the edge message
		Me_pre = self.prepare_message(V, E, K) # Z x N x K x d_model

		# process the message
		Me = self.edge_messenger(Me_pre) * edge_mask.unsqueeze(3) # Z x N x K x d_model

		if t is None:

			# update the edges
			E = self.edge_messenger_norm(E + self.dropout(Me)) # Z x N x K x d_model

		else:

			# update the edges with timestep conditioning
			E = self.edge_messenger_norm(E + self.dropout(Me), t) # Z x N x K x d_model

		return V, E

	def prepare_message(self, V, E, K):

		dimZ, dimN, d_model = V.shape
		_, _, dimK = K.shape

		# gather neighbor nodes
		Vi = V.unsqueeze(2).expand(-1,-1,dimK,-1) # Z x N x K x d_model
		Ki = K.unsqueeze(3).expand(-1,-1,-1,d_model) # Z x N x K x d_model
		Vj = torch.gather(Vi, 1, Ki) # Z x N x K x d_model

		Mv_pre = torch.cat([Vi, Vj, E], dim=3)

		return Mv_pre

class StaticLayerNorm(nn.Module):
	'''just normalizes each token to have a mean of 0 and var of 1, no scaling and shifting'''
	def __init__(self, d_model):
		super(StaticLayerNorm, self).__init__()
		self.d_model = d_model
	def forward(self, x):
		centered = x - x.mean(dim=-1, keepdim=True) 
		std = centered.std(dim=-1, keepdim=True)
		std = std.masked_fill(std==0, 1)
		return centered / std

class adaLN(nn.Module):
	'''adaptive layer norm to perform affine transformation conditioned on timestep'''
	def __init__(self, d_model=512, dropout=0.0):
		super(adaLN, self).__init__()
		self.gamma_beta = MLP(d_in=d_model, d_out=2*d_model, d_hidden=d_model, hidden_layers=0, dropout=dropout, act="silu", zeros=False)
		self.norm = StaticLayerNorm(d_model)
	def forward(self, x, e_t):
		gamma_beta = self.gamma_beta(e_t)
		gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
		# et is Z x N x D, but sometimes norming the edges, so expand in neighbor dim
		if x.dim()==4:
			gamma = gamma.unsqueeze(2)
			beta = beta.unsqueeze(2)
		x = gamma*self.norm(x) + beta
		return x

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
