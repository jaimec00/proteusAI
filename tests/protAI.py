import torch
from torch.nn.functional import one_hot as onehot
from proteusAI import proteusAI

def main():
	
	device = torch.device("cuda")


	# init inputs
	batch, N, d_model, nheads = 1, 16384, 512, 16

	coords = 20 * torch.rand((batch, N, 3), device=device, dtype=torch.float32)
	aa_labels = torch.randint(0, 20, (batch, N), device=device, dtype=torch.long)
	aas = onehot(aa_labels, num_classes=21).to(torch.float32)
	mask = torch.rand((batch, N), device=device, dtype=torch.float32) > 1
	
	# init the model
	model = proteusAI(	d_model=d_model, 
						min_wl=3.7, max_wl=20, base_wl=20, 
						d_hidden_wl=1024, hidden_layers_wl=0, 

						# aa mlp
						d_hidden_aa=1024, hidden_layers_aa=0, esm2_weights_path="utils/model_utils/esm2/esm2_t33_650M_UR50D.pt",

						# geometric attn + ffn
						encoder_layers=1,
						n_head=16,
						min_spread=3.7, max_spread=7, base_spreads=20, num_spread=64,
						min_rbf=0.01, max_rbf=0.99,

						d_hidden_attn=1024, hidden_layers_attn=0,
						
						# dropout
						dropout=0.00,
					) # default args
	model.to(device)

	# warmup
	# for i in range(3):
	# 	# forward pass
	# 	out = model(coords, aas, mask)
		
	# 	# backward pass
	# 	out.sum().backward()

	# empty cache
	torch.cuda.empty_cache()
	torch.cuda.synchronize()

	# start profiler
	torch.cuda.cudart().cudaProfilerStart()

	# forward pass
	out = model(coords, aas, mask)
	
	# backward pass
	out.sum().backward()

	# stop profiler
	torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
	main()
