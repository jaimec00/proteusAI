
import torch
import torch.nn as nn

from utils.model_utils.wf_embedding.wf_embedding import WaveFunctionEmbedding
from data.constants import alphabet

class FeaturizeProtein(nn.Module):
	def __init__(self, K=30, d_model=128, min_wl=3.5, max_wl=25.0, base_wl=20.0, learn_wl=True, min_rbf=2.0, max_rbf=22.0, num_rbfs=16):
		super(FeaturizeProtein, self).__init__()

		self.K = K
		
		self.wf_embedding = WaveFunctionEmbedding(d_wf=d_model, min_wl=min_wl, max_wl=max_wl, base_wl=base_wl, learn_wl=learn_wl)

		self.node_norm = nn.LayerNorm(d_model)
		self.node_proj = nn.Linear(d_model, d_model)

		self.edge_norm = nn.LayerNorm(int(num_rbfs*4*4))
		self.edge_proj = nn.Linear(int(num_rbfs*4*4), d_model)
		
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, int(num_rbfs)))
		self.spread = (max_rbf - min_rbf) / num_rbfs

	def forward(self, C, L, chain_idxs, node_mask=None): # C is Z x N x 3[N,Ca,C] x 3[x,y,z]

		# get virtual Cb coords
		C, Ca, Cb = self.get_coords(C, chain_idxs, norm=False)

		# get nodes
		V = self.get_nodes(Ca, Cb, L, node_mask)

		# get neighbors
		K, edge_mask = self.get_neighbors(Ca, node_mask=node_mask) # Z x N x K

		# get initial edges
		E = self.get_edges(C, K) # Z x N x K x De

		return V, E, K, edge_mask

	def get_coords(self, C, chain_idxs, norm=False):
		
		# get Cb coords
		Ca, Cb = self.wf_embedding.get_CaCb_coords(C, chain_idxs, norm=False) # Z x N x 3

		# add the Cb coords to the tensor
		C = torch.cat([C, (Ca + Cb).unsqueeze(2)], dim=2) # Z x N x 4[N,Ca,C,Cb] x 3

		return C, Ca, Cb

	def get_nodes(self, Ca, Cb, L, node_mask):

		# embed the nodes with wf embedding, includes seq info
		V = self.wf_embedding(Ca, Cb, L, key_padding_mask=node_mask) # Z x N x Dw
		V = self.node_proj(self.node_norm(V)) # Z x N x Dv

		return V

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

		# norm and project to Z x N x K x d_model
		E = self.edge_proj(self.edge_norm(E))

		return E