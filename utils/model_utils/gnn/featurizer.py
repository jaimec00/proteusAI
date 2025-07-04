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
	def __init__(self, 	K=30, d_model=128, 
						min_wl=3.5, max_wl=25.0, base_wl=20.0, anisotropic=True, learn_wl=True, learn_aa=False, 
						min_rbf=2.0, max_rbf=22.0, num_rbfs=16, max_dist=float("inf")):
		super(FeaturizeProtein, self).__init__()

		self.K = K
		self.max_dist = max_dist
		
		self.wf_embedding = WaveFunctionEmbedding(d_wf=d_model, min_wl=min_wl, max_wl=max_wl, base_wl=base_wl, anisotropic=anisotropic, learn_wl=learn_wl, learn_aa=learn_aa)

		self.node_norm = nn.LayerNorm(d_model)
		self.node_proj = nn.Linear(d_model, d_model)

		self.edge_norm = nn.LayerNorm(int(num_rbfs*4*4))
		self.edge_proj = nn.Linear(int(num_rbfs*4*4), d_model)

		self.seq_proj = nn.Linear(len(alphabet), d_model)
		
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, int(num_rbfs)))
		self.spread = (max_rbf - min_rbf) / num_rbfs

	def forward(self, C, S, chain_idxs, node_mask=None): # C is Z x N x 3[N,Ca,C] x 3[x,y,z]

		# get Cb coords
		Ca, Cb = self.wf_embedding.get_CaCb_coords(C, chain_idxs, norm=False) # Z x N x 3

		# embed the nodes with wf embedding
		V = self.wf_embedding(Ca, Cb, key_padding_mask=node_mask) # Z x N x Dw
		V = self.node_proj(self.node_norm(V)) # Z x N x Dv

		# add the Cb coords to the tensor
		C = torch.cat([C, (Ca + Cb).unsqueeze(2)], dim=2) # Z x N x 4[N,Ca,C,Cb] x 3

		# get neighbors
		K, edge_mask = self.get_neighbors(Ca, node_mask=node_mask) # Z x N x K

		# get initial edges
		E = self.get_edges(C, K) # Z x N x K x De

		# featurize the sequence and format it with edges
		S = self.featurize_seq(S)

		return V, E, K, S, edge_mask

	def featurize_seq(self, S):
		'''S is Z x N labels'''

		# turn into onehot tensor, and zero out masked positions
		no_seq = S == -1
		S = torch.nn.functional.one_hot(torch.where(no_seq, 0, S), num_classes=len(alphabet)).to(torch.float32) # Z x N x alphabet

		# featurize, 0 where labels==-1
		S = (~no_seq).unsqueeze(2) * self.seq_proj(S) # Z x N x De

		return S

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
		edge_mask = edge_mask & (topk.values!=0) & (topk.values < self.max_dist) # exclude self and distant neighbors
		topk = torch.where(edge_mask, topk.indices, node_idxs) # Z x N x K

		return topk, edge_mask
		
	def get_edges(self, C, K):

		dimZ, dimN, dimA, dimS = C.shape
		_, _, dimK = K.shape
		
		# get the coords for the neighbors
		CK = torch.gather(C.unsqueeze(2).expand(-1,-1,dimK,-1,-1), 1, K.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,dimA,dimS)) # Z x N x K x 4[N,Ca,C,Cb] x 3[x,y,z]

		# get neighbor distances
		dists = torch.sqrt(torch.sum((C.unsqueeze(2).unsqueeze(4) - CK.unsqueeze(3))**2, dim=5)) # Z x N x 1 x 4 x 1 x 3 - # Z x N x K x 1 x 4 x 3 --> # Z x N x K x 4 x 4

		# compute rbfs
		rbfs = torch.exp(-((dists.unsqueeze(5) - self.rbf_centers.view(1,1,1,1,1,-1))**2) / (self.spread**2)) # Z x N x K x 4 x 4 x num_spreads

		# flatten to Z x N x K x (4*4*num_spreads)
		E = rbfs.view(dimZ, dimN, dimK, -1)

		# norm and project to Z x N x K x De
		E = self.edge_proj(self.edge_norm(E))

		return E
