# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein based on backbone coordinates. 
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
# from torch.utils.checkpoint import checkpoint

from utils.model_utils.gnn.gnn import FeaturizeProtein, GNNEncoder, GNNDecoder
from data.constants import canonical_aas

# ----------------------------------------------------------------------------------------------------------------------

class proteusAI(nn.Module):
	def __init__(self,  K=30, De=128, Dw=128, Dv=128, Ds=128, # model dims
						min_wl=3.5, max_wl=12.0, base_wl=20.0, anisotropic=True, learn_wl=True, learn_aa=False, # node embedding
						min_rbf=2.0, max_rbf=22.0, num_rbfs=16, # edge embedding
						enc_layers=3, dec_layers=3, dropout=0.00 # general
					):

		super(proteusAI, self).__init__()

		self.featurizer = FeaturizeProtein( K=K, De=De, Dw=Dw, Dv=Dv, # model dims
											min_wl=min_wl, max_wl=max_wl, base_wl=base_wl, anisotropic=anisotropic, learn_wl=learn_wl, learn_aa=learn_aa, # node features (wf embedding)
											min_rbf=min_rbf, max_rbf=max_rbf, num_rbfs=num_rbfs # edge features
										)

		self.encoders = nn.ModuleList([GNNEncoder(De=De, Dv=Dv, dropout=dropout) for _ in range(enc_layers)])
		self.decoders = nn.ModuleList([GNNDecoder(De=De, Dv=Dv, dropout=dropout) for _ in range(dec_layers)])

		self.out_proj = nn.Linear(Dv, len(canonical_aas))

	def forward(self, C, L, chain_idxs, node_mask=None, decoding_order=None, inference=False, temp=1e-6):
		'''
		C is coords
		L is labels (-1 means to be decoded, unless nodemask is true at that position)
		'''

		# featurize protein
		V, E, K, S, edge_mask, autoregressive_mask = self.featurizer(C, L, chain_idxs, node_mask, decoding_order)

		# encode structure
		V, E = self.encode(V, E, K, edge_mask)

		# decode sequence
		if inference:
			V = self.inference(V, E, K, S, L, edge_mask, decoding_order, autoregressive_mask, temp)
		else:
			V = self.decode(V, E, K, S, edge_mask, autoregressive_mask)

		return V

	def encode(self, V, E, K, edge_mask):

		for encoder in self.encoders:
			V, E = encoder(V, E, K, edge_mask)

		return V, E

	def decode(self, V, E, K, S, edge_mask, autoregressive_mask):

		V_new = V # use this to gather original nodes, not updated by neighbor seqs for autoregressive training
		for decoder in self.decoders:
			V_new = decoder(V_new, V, E, K, S, edge_mask, autoregressive_mask)

		V = self.out_proj(V_new)

		return V

	def inference(self, V, E, K, S, L, edge_mask, decoding_order, autoregressive_mask, temp=1e-6):

		# initialize boolean mask of fixed positions (where L != -1)
		fixed = L!=-1 # Z x N

		# skip nodes that need not be decoded (masked nodes in edge_mask have all edges=False)
		decoded = edge_mask.all(dim=2) | fixed # Z x N

		# keep track of the current position to be decoded
		decoding_position = 0

		# done when all nodes have been decoded
		while not decoded.all():

			# decode the sequence, new var name for nodes so decoder always starts with original encoder nodes
			V_new = self.decode(V, E, K, S, edge_mask, autoregressive_mask) # Z x N x len(canonical_aas)

			# sample an amino acid
			L_sampled = self.sample(V_new, temp) # Z x N

			# decide which positions to update if it is not fixed
			update_seq = (decoding_order == decoding_position) & (~decoded) # Z x N 

			# update the labels 
			L[update_seq] = L_sampled[update_seq] # Z x N

			# featurize the updated sequence for the next iteration
			S = self.featurizer.featurize_seq(L) # Z x N x De
			
			# update the decoded mask
			decoded |= update_seq # Z x N

			# increment decoding position for next iteration
			decoding_position += 1

		return L # Z x N

	def sample(self, V, temp=1e-6):

		# convert to aa probabilities
		probs = torch.softmax(V / temp, dim=2) # Z x N x len(canonical_aas)

		# sample from the distribution
		sample = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1)) # Z x N

		return sample