import torch
from torch.nn.functional import one_hot as onehot
import proteusAI

def main():
    
    device = torch.device("cuda")


    # init inputs
    batch, N, d_model, nheads = 1, 16384, 512, 16

    coords = 20 * torch.rand((batch, N, 3), device=device, dtype=torch.float32)
    aa_labels = torch.randint(0, 20, (batch, N), device=device, dtype=torch.long)
    aas = onehot(aa_labels, num_classes=21) 
    mask = torch.rand((batch, N), device=device, dtype=torch.bool) < 1
    
    # init the model
    model = proteusAI(d_model=d_model, n_head=16) # default args
    model = model.to(device)

    # start profiler
	torch.cuda.cudart().cudaProfilerStart()

    # forward pass
    out = model(coords, aas, mask)
    
    # backward pass
    out.sum().backward()

    # stop profiler
	torch.cuda.cudart().cudaProfilerStart()


if __name__ == "__main__":
    main()
