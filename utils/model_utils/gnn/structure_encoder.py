
import torch
import torch.nn as nn

from utils.model_utils.base_modules.base_modules import MPNN

class StructureEncoder(nn.Module):
	def __init__(self, d_model=128, layers=3, dropout=0.00):
		super().__init__()

		self.encoders = nn.ModuleList([MPNN(d_model=d_model, dropout=dropout) for _ in range(layers)])

	def forward(self, V, E, K, edge_mask):

		for encoder in self.encoders:
			V, E = encoder(V, E, K, edge_mask)

		return V, E