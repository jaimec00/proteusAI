'''
my goal here is to copy protein mpnn, except for a few additions/modifications to their network
it seems like all to all attention does not work well, get my best performance with masking,
suggesting gnn are better suited for this task, or maybe im just incompetent, we'll see

so pmpnn had nodes and edges
nodes start out as all zeros, but i will try to make the nodes the "global" view of the structure via wave function embedding
edges are pairwise distances encoded as rbfs between each inter-residue atom pair

nodes is easy, edges is easy to encode, but i would like to incorporate sparse attention on the nearest neighbors to update these

it will also be easier to incorporate sequence info if i am doing autoregressive, and it is also helpful that they only incorporate it in the decoder

'''
import torch
import torch.nn as nn

from utils.model_utils.base_modules.base_modules import MLP
from utils.model_utils.wf_embedding.wf_embedding import WaveFunctionEmbedding
from data.constants import alphabet

class FeaturizeProtein(nn.Module):
	def __init__(self, K=30, De=128, Dw=128, Dv=128, min_wl=3.5, max_wl=25.0, base_wl=20.0, anisotropic=True, learn_wl=True, learn_aa=False, min_rbf=2.0, max_rbf=22.0, num_rbfs=16):
		super(FeaturizeProtein, self).__init__()

		self.K = K
		
		self.wf_embedding = WaveFunctionEmbedding(d_wf=Dw, min_wl=min_wl, max_wl=max_wl, base_wl=base_wl, anisotropic=anisotropic, learn_wl=learn_wl, learn_aa=learn_aa)

		self.node_norm = nn.LayerNorm(Dw)
		self.node_proj = nn.Linear(Dw, Dv)

		self.edge_norm = nn.LayerNorm(int(num_rbfs*4*4))
		self.edge_proj = nn.Linear(int(num_rbfs*4*4), De)

		self.seq_proj = nn.Linear(len(alphabet), De)
		
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, int(num_rbfs)))
		self.spread = (max_rbf - min_rbf) / num_rbfs

	def forward(self, C, S, chain_idxs, node_mask=None, decoding_order=None): # C is Z x N x 3[N,Ca,C] x 3[x,y,z]

		# get Cb coords
		Ca, Cb = self.wf_embedding.get_CaCb_coords(C, chain_idxs, norm=False) # Z x N x 3

		# embed the nodes with wf embedding
		V = self.wf_embedding(Ca, Cb, key_padding_mask=node_mask) # Z x N x Dw
		# V = torch.zeros(C.size(0), C.size(1), 128, device=C.device) # Z x N x Dw # try with all zeros
		V = self.node_proj(self.node_norm(V)) # Z x N x Dv

		# add the Cb coords to the tensor
		C = torch.cat([C, (Ca + Cb).unsqueeze(2)], dim=2) # Z x N x 4[N,Ca,C,Cb] x 3

		# get neighbors
		K, edge_mask = self.get_neighbors(Ca, node_mask=node_mask) # Z x N x K

		# get initial edges
		E = self.get_edges(C, K) # Z x N x K x De

		# featurize the sequence and format it with edges
		S = self.featurize_seq(S)

		# get the autoregressive mask
		autoregressive_mask = self.get_autoregressive_mask(K, decoding_order)

		return V, E, K, S, edge_mask, autoregressive_mask

	def featurize_seq(self, S):
		'''S is Z x N labels'''

		# turn into onehot tensor, and zero out masked positions
		no_seq = S == -1
		S = torch.nn.functional.one_hot(torch.where(no_seq, 0, S), num_classes=len(alphabet)).to(torch.float32) # Z x N x alphabet

		# featurize, 0 where labels==-1
		S = (~no_seq).unsqueeze(2) * self.seq_proj(S) # Z x N x De

		return S

	def get_autoregressive_mask(self, K, decoding_order):
		'''
		decoding order should be a Z x N tensor, with each element corresponding to the order it will be decoded
		easy way to prepare it for training is 

		decoding_order = torch.argsort( torch.where(node_mask, torch.rand((Z, N)), float("inf")), dim=1 ).indices

		'''

		decoding_order = decoding_order.unsqueeze(2) # Z x N x 1

		neighbor_decoding_order = torch.gather(decoding_order.expand(-1,-1,K.size(2)), 1, K) # Z x N x K

		autoregressive_mask = neighbor_decoding_order < decoding_order # Z x N x K

		return autoregressive_mask

	def get_neighbors(self, Ca, node_mask=None):

		dimZ, dimN, dimS = Ca.shape
		assert dimN>=self.K

		# get distances
		dists = torch.sqrt(torch.sum((Ca.unsqueeze(1) - Ca.unsqueeze(2))**2, dim=3)) # Z x N x N
		dists = torch.where(dists==0 | node_mask.unsqueeze(2), float("inf"), dists) # Z x N x N
		
		# get topk 
		topk = dists.topk(self.K, dim=2, largest=False) # Z x N x K

		# masked nodes have themselves as edges, masked edges are the corresponding node
		node_idxs = torch.arange(dimN, device=dists.device).view(1,-1,1) # 1 x N x 1
		edge_mask = ~(node_mask.unsqueeze(2) | torch.gather(node_mask.unsqueeze(2).expand(-1,-1,self.K), 1, topk.indices))
		edge_mask = edge_mask & (topk.values!=0) & (topk.values < 12) # exclude self and distant neighbors
		topk = torch.where(edge_mask, topk.indices, node_idxs) # Z x N x K

		return topk, edge_mask
		
	def get_edges(self, C, K):

		dimZ, dimN, dimA, dimS = C.shape
		_, _, dimK = K.shape
		
		# get the coords for the neighbors
		CK = torch.gather(C.unsqueeze(2).expand(-1,-1,dimK,-1,-1), 1, K.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,dimA,dimS)) # Z x N x K x 4[N,Ca,C,Cb] x 3[x,y,z]

		# get neighbor distances
		dists = torch.sqrt(torch.sum((C.unsqueeze(2).unsqueeze(4) - CK.unsqueeze(3))**2, dim=5)) # Z x N x 1 x 4 x 1 x 3 - # Z x N x K x 1 x 4 x 3 --> # Z x N x K x 4 x 4

		# compute rbfs (i originally had fixed centers at zero, but realized pmpnn had linearly spaced centers with fixed spreads, so trying that instead)
		rbfs = torch.exp(-((dists.unsqueeze(5) - self.rbf_centers.view(1,1,1,1,1,-1))**2) / (self.spread**2)) # Z x N x K x 4 x 4 x num_spreads

		# flatten to Z x N x K x (4*4*num_spreads)
		E = rbfs.view(dimZ, dimN, dimK, -1)

		# norm and project to Z x N x K x De
		E = self.edge_proj(self.edge_norm(E))

		return E

class GNN(nn.Module):
	'''
	just a couple methods used by enc/dec
	'''

	def __init__(self):
		super(GNN, self).__init__()

	def gather_nodes(self, V, K):

		dimZ, dimN, dimDv = V.shape
		_, _, dimK = K.shape

		# gather neighbor nodes
		Vi = V.unsqueeze(2).expand(-1,-1,dimK,-1) # Z x N x K x Dv
		Ki = K.unsqueeze(3).expand(-1,-1,-1,dimDv) # Z x N x K x Dv
		Vj = torch.gather(Vi, 1, Ki) # Z x N x K x Dv

		return Vi, Vj
	
	def gather_seq(self, S, K):

		dimZ, dimN, dimK = K.shape # Z x N x K
		_, _, dimDe = S.shape # Z x N x De

		# gather neighbor nodes
		S = S.unsqueeze(2).expand(-1,-1,dimK,-1) # Z x N x K x De
		K = K.unsqueeze(3).expand(-1,-1,-1,dimDe) # Z x N x K x De
		S = torch.gather(S, 1, K) # Z x N x K x De

		return S

class GNNEncoder(GNN):
	def __init__(self, De=128, Dv=128, dropout=0.00):
		super(GNNEncoder, self).__init__()

		self.node_messenger = MLP(d_in=2*Dv+De, d_out=Dv, d_hidden=Dv, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
		self.node_messenger_norm = nn.LayerNorm(Dv)

		self.ffn = MLP(d_in=Dv, d_out=Dv, d_hidden=Dv*4, hidden_layers=0, dropout=dropout, act="gelu", zeros=False)
		self.ffn_norm = nn.LayerNorm(Dv)

		self.edge_messenger = MLP(d_in=2*Dv+De, d_out=De, d_hidden=De, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
		self.edge_messenger_norm = nn.LayerNorm(De)

		self.dropout = nn.Dropout(dropout)

	def forward(self, V, E, K, edge_mask=None):
		'''
		for now copying pmpnn, except node features are initialized with wf embedding instead of zeros. 
		will then try attention w/ receiving nodes as Q, edges as K, and sending nodes as V for the node messenger
		edge updates will be the same as in pmpnn, since dont need attention since only depends on the nodes
		'''

		# gathe neighbor nodes
		Vi, Vj = self.gather_nodes(V, K) # Z x N x K x Dv

		# cat the node and edge tensors to create the message
		Mv_pre = torch.cat([Vi, Vj, E], dim=3) # Z x N x K x (2*Dv + De)

		# process the message
		Mv = self.node_messenger(Mv_pre) * edge_mask.unsqueeze(3) # Z x N x K x Dv

		# send the message
		V = self.node_messenger_norm(V + self.dropout(Mv.sum(dim=2))) # Z x N x Dv

		# process the updated node
		V = self.ffn_norm(V + self.dropout(self.ffn(V)))

		# update the edges from new nodes
		Vi, Vj = self.gather_nodes(V, K) # Z x N x K x Dv

		# prepare message
		Me_pre = torch.cat([Vi, Vj, E], dim=3) # Z x N x K x (2*Dv + De)

		# process the message
		Me = self.edge_messenger(Me_pre) * edge_mask.unsqueeze(3) # Z x N x K x De

		# update the edges
		E = self.edge_messenger_norm(E + self.dropout(Me)) # Z x N x K x Dv

		return V, E


class GNNDecoder(GNN):
	def __init__(self, De=128, Dv=128, dropout=0.00):
		super(GNNDecoder, self).__init__()

		self.node_messenger = MLP(d_in=2*De + 2*Dv, d_out=Dv, d_hidden=Dv, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
		self.node_messenger_norm = nn.LayerNorm(Dv)

		self.ffn = MLP(d_in=Dv, d_out=Dv, d_hidden=4*Dv, hidden_layers=0, dropout=dropout, act="gelu", zeros=False)
		self.ffn_norm = nn.LayerNorm(Dv)

		self.dropout = nn.Dropout(dropout)

	def forward(self, V_new, V_old, E, K, S, edge_mask=None, autoregressive_mask=None):
		'''
		S is Z x N x De of sequence embeddings
		'''

		# gather the sequence features, mask out future neighbos
		Es = self.gather_seq(S, K) * autoregressive_mask.unsqueeze(3) # Z x N x K x De

		# cat with edge features
		E = torch.cat([E, Es], dim=3) # Z x N x K x (2*De)

		# gather i and j nodes. updated nodes are for previously decoded neighbors, old nodes are for future neighbors to prevent information leakage
		Vi, Vj_new = self.gather_nodes(V_new, K) # Z x N x K x Dv
		_, Vj_old = self.gather_nodes(V_old, K)

		# Vj now has old nodes for future neighbors, and new nodes for decoded neighbors
		Vj = torch.where(autoregressive_mask.unsqueeze(3), Vj_new, Vj_old)

		# prepare the message
		Mv_pre = torch.cat([Vi, Vj, E], dim=3) # Z x N x K x (2*De + 2*Dv)

		# process the message
		Mv = self.node_messenger(Mv_pre) * edge_mask.unsqueeze(3) # Z x N x K x Dv

		# send the message
		V = self.node_messenger_norm(V_new + self.dropout(Mv.sum(dim=2))) # Z x N x Dv

		# process the node
		V = self.ffn_norm(V + self.dropout(self.ffn(V))) # Z x N x Dv

		return V	
