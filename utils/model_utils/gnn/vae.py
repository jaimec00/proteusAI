

import torch
import torch.nn as nn

from utils.model_utils.base_modules.base_modules import MLP, MPNN
from data.constants import canonical_aas

class SequenceEncoder(nn.Module):
	def __init__(self, d_model=128, layers=3, dropout=0.00):
		super().__init__()

		self.sequence_messenger = MLP(d_in=2*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu")
		self.sequence_messenger_norm = nn.LayerNorm(d_model)

		self.encoders = nn.ModuleList([MPNN(d_model, dropout=dropout) for _ in range(layers)])

		self.latent_params = MLP(d_in=d_model, d_out=2*d_model, d_hidden=4*d_model, hidden_layers=0, dropout=dropout, act="gelu")

		self.dropout = nn.Dropout(dropout)

	def forward(self, V, E, K, S, edge_mask):

		# combine structure node embedding and sequence embeddings
		VS = torch.cat([V, S], dim=2)
		V = self.sequence_messenger_norm(V + self.dropout(self.sequence_messenger(VS)))

		# perform message passing
		for encoder in self.encoders:
			V, E = encoder(V, E, K, edge_mask)

		# get params and split into mu and log var
		mu_logvar = self.latent_params(V)
		Z_mu, Z_logvar = torch.chunk(mu_logvar, dim=2, chunks=2)

		# sample a latent sequence vector with structure context
		Z = Z_mu + torch.randn_like(S)*torch.exp(0.5*Z_logvar)

		return Z, Z_mu, Z_logvar

class SequenceDecoder(nn.Module):
	def __init__(self, d_model=128, layers=3, dropout=0.00):
		super().__init__()

		self.sequence_messenger = MLP(d_in=2*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu")
		self.sequence_messenger_norm = nn.LayerNorm(d_model)

		self.decoders = nn.ModuleList([MPNN(d_model, dropout=dropout) for _ in range(layers)])

		self.out_proj = nn.Linear(d_model, len(canonical_aas))

		self.dropout = nn.Dropout(dropout)


	def forward(self, V, E, K, Z, edge_mask, sample=False, temp=1e-6):

		# combine structure node embedding and sequence embeddings
		VZ = torch.cat([V, Z], dim=2)
		V = self.sequence_messenger_norm(V + self.dropout(self.sequence_messenger(VZ)))

		# perform message passing
		for decoder in self.decoders:
			V, E = decoder(V, E, K, edge_mask)

		# convert to aa logits
		S = self.out_proj(V)

		if sample:
			S = self.sample(S, temp)

		return S

	def sample(self, S, temp):

		probs = torch.softmax(S/temp, dim=2)
		sample = torch.multinomial(probs.view(-1, probs.size(2)), 1).view(probs.size(0), probs.size(1))

		return sample