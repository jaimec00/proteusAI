import torch
import torch.nn as nn

from utils.model_utils.base_modules.base_modules import MLP
from utils.model_utils.wf_embedding.wf_embedding import WaveFunctionEmbedding

class MPNN(nn.Module):
	def __init__(self, d_model=128, dropout=0.00, update_edge=True):
		super(MPNN, self).__init__()

		self.node_messenger = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
		self.node_messenger_norm = nn.LayerNorm(d_model)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_model*4, hidden_layers=0, dropout=dropout, act="gelu", zeros=False)
		self.ffn_norm = nn.LayerNorm(d_model)

		self.update_edge = update_edge
		if self.update_edge:
			self.edge_messenger = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
			self.edge_messenger_norm = nn.LayerNorm(d_model)

		self.dropout = nn.Dropout(dropout)

	def forward(self, V, E, K, edge_mask=None):
		'''
		for now copying pmpnn, except node features are initialized with wf embedding instead of zeros. 
		will then try attention w/ receiving nodes as Q, edges as K, and sending nodes as V for the node messenger
		edge updates will be the same as in pmpnn, since dont need attention since only depends on the nodes
		'''

		# gathe neighbor nodes
		Mv_pre = self.prepare_message(V, E, K) # Z x N x K x d_model

		# process the message
		Mv = self.node_messenger(Mv_pre) * edge_mask.unsqueeze(3) # Z x N x K x d_model

		# send the message
		V = self.node_messenger_norm(V + self.dropout(Mv.sum(dim=2))) # Z x N x d_model

		# process the updated node
		V = self.ffn_norm(V + self.dropout(self.ffn(V)))

		if self.update_edge:

			# update the edges from new nodes
			Me_pre = self.prepare_message(V, E, K) # Z x N x K x d_model

			# process the message
			Me = self.edge_messenger(Me_pre) * edge_mask.unsqueeze(3) # Z x N x K x d_model

			# update the edges
			E = self.edge_messenger_norm(E + self.dropout(Me)) # Z x N x K x d_model

		return V, E

	def prepare_message(self, V, E, K):

		dimZ, dimN, d_model = V.shape
		_, _, dimK = K.shape

		# gather neighbor nodes
		Vi = V.unsqueeze(2).expand(-1,-1,dimK,-1) # Z x N x K x d_model
		Ki = K.unsqueeze(3).expand(-1,-1,-1,d_model) # Z x N x K x d_model
		Vj = torch.gather(Vi, 1, Ki) # Z x N x K x d_model

		# cat the node and edge tensors to create the message
		Mv_pre = torch.cat([Vi, Vj, E], dim=3) # Z x N x K x 3*d_model

		return Mv_pre