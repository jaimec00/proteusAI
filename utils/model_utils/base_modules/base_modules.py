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



class ConvFormer(nn.Module):
	def __init__(self, kernel_sizes=(3,5,7), d_model=256, dropout=0.0):
		super(ConvFormer, self).__init__()
		self.conv_weights = nn.ParameterList([nn.Parameter(torch.randn((d_model, d_model, kernel_size))) for kernel_size in kernel_sizes])
		self.conv_biases = nn.ParameterList([nn.Parameter(torch.randn((d_model))) for kernel_size in kernel_sizes])
		self.mlp = MLP(d_in=d_model, d_out=d_model, d_hidden=4*d_model, hidden_layers=0, dropout=dropout) 
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, chain_idxs):

		B, L, _ = x.shape

		chain_tensor  = torch.zeros(x.size(0), x.size(1), dtype=torch.int32, device=x.device)
		for sample_idx, sample in enumerate(chain_idxs):
			for sample_chain_idx, (chain_start, chain_stop) in enumerate(sample, start=1): # start at one so that padded positions have idx=0, ie not mixed with nonpadded
				chain_tensor[sample_idx, chain_start:chain_stop] = sample_chain_idx

		x1 = x.permute(0,2,1)
		for conv_weight, conv_bias in zip(self.conv_weights, self.conv_biases):

			cin, cout, K = conv_weight.shape

			# Compute output length O = ceil(L / stride)
			O = L
			# Compute total padding
			P_total = K-1
			if P_total < 0:
				P_total = 0
			pad_left = P_total // 2
			pad_right = P_total - pad_left

			# Pad x along length
			# F.pad works for 3D input: pad=(pad_left, pad_right) on last dim
			x_padded = torch.nn.functional.pad(x1, (pad_left, pad_right))

			# Pad idxs along length with pad_value
			# Build a new tensor of shape (B, L + pad_left + pad_right)
			L_padded = L + pad_left + pad_right
			device = x.device
			dtype_idxs = chain_tensor.dtype
			idxs_padded = torch.full((B, L_padded), 0, dtype=dtype_idxs, device=device)
			# copy original idxs into center
			idxs_padded[:, pad_left: pad_left + L] = chain_tensor

			# Extract windows via unfold
			# x_padded.unfold -> shape (B, cin, O, K)
			Xw = x_padded.unfold(dimension=2, size=K, step=1)
			# idxs_padded.unfold -> shape (B, O, K)
			idxs_windows = idxs_padded.unfold(dimension=1, size=K, step=1)

			# Build mask: keep positions where idx == center idx
			center = K // 2
			center_idxs = idxs_windows[:, :, center]        # shape (B, O)
			mask = (idxs_windows == center_idxs.unsqueeze(2))  # shape (B, O, K), bool

			# Apply mask: broadcast over cin
			# Xw: (B, cin, O, K); mask.unsqueeze(1): (B,1,O,K) -> broadcast to (B,cin,O,K)
			Xw_masked = Xw * mask.unsqueeze(1)

			# Contract with weight via einsum
			# weight shape: (cout, cin, K)
			# out[b, o, i] = sum_c,k Xw_masked[b, c, i, k] * weight[o, c, k]
			out = torch.einsum('b c i k, o c k -> b o i', Xw_masked, conv_weight)
			# Add bias if provided
			out = out + conv_bias.view(1, cout, 1)

			x1 = out

		x = self.norm1(x + self.dropout(x1.permute(0,2,1)))
		x = self.norm2(x + self.dropout(self.mlp(x)))

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
