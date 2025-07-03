
class GraphAttention(nn.Module):
	def __init__(self, d_model, heads=4):
		super(GraphAttention, self).__init__()

		self.heads = heads
		self.d_model = d_model

		if self.d_model % self.heads != 0: raise ValueError(f"number of dimensions ({self.d_model}) must be divisible by number of attention heads ({self.heads})")
		self.d_k = self.d_model // self.heads

		# QKV projection weight and bias matrices

		# init xavier distribution
		xavier_scale = (6/(self.d_k + d_model))**0.5

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k

		self.q_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k
		self.k_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k
		self.v_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

		self.Mv_norm = nn.LayerNorm(d_model)
		self.V_norm = nn.LayerNorm(d_model)

	def forward(self, V, Mv, edge_mask):
		'''
		nodes should be q, and message should be k and v
		V in Z x N x Dv
		Mv in Z x N x K x Dv
		edge_mask in Z x N x K
		'''

		V = self.V_norm(V)
		Mv = self.Mv_norm(Mv)

		Z, N, _ = V.shape

		# project the tensors
		VQ = (torch.matmul(V.unsqueeze(1), self.q_proj.unsqueeze(0)) + self.q_bias.unsqueeze(0).unsqueeze(2)).unsqueeze(3) # Z x 1 x N x Dv @ 1 x H x Dv x Dk --> Z x H x N x 1 x Dk
		MvK = torch.matmul(Mv.unsqueeze(1), self.k_proj.unsqueeze(0).unsqueeze(2)) + self.k_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3) # Z x 1 x N x K x Dv @ 1 x H x 1 x Dv x Dk --> Z x H x N x K x Dk
		MvV = torch.matmul(Mv.unsqueeze(1), self.v_proj.unsqueeze(0).unsqueeze(2)) + self.v_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3) # Z x 1 x N x K x Dv @ 1 x H x 1 x Dv x Dk --> Z x H x N x K x Dk

		# attention
		S = torch.matmul(VQ, MvK.transpose(3,4)) / (self.d_k**0.5) # Z x H x N x 1 x K 
		S = torch.where(edge_mask.unsqueeze(1).unsqueeze(3), -float("inf"), S)
		P = torch.softmax(S, dim=4)
		out = torch.matmul(P, MvV).squeeze(3) # Z x H x N x Dk

		# cat heads
		out = out.permute(0,2,3,1) # batch x N x d_k x heads
		out = out.reshape(Z, N, self.d_model) # batch x N x d_k x heads --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model

