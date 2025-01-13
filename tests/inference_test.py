import torch
from proteusAI import proteusAI

def main():

    with torch.no_grad():
        batch, N, d_model = 2, 8, 512
        temp = 0.1
        device = torch.device("cuda")

        coords = torch.randn((batch, N, 3), dtype=torch.float32, device=device)
        aas = torch.full((batch, N, 20), 1/20, dtype=torch.float32, device=device)
        mask = torch.zeros(batch, N, dtype=torch.bool, device=device)

        model = proteusAI(	d_model=d_model,
                            min_wl=3.7, max_wl=20, base_wl=20, 
                            d_hidden_wl=1024, hidden_layers_wl=0, 
                            d_hidden_aa=1024, hidden_layers_aa=0,
                            dualcoder_layers=4,
                            n_head=4,
                            min_spread=3.7, max_spread=7, base_spreads=20, 
                            min_rbf=0.05, max_rbf=0.99, 
                            d_hidden_attn=1024, hidden_layers_attn=0,						
                            dropout=0.00,
                            include_ncaa=False
                    )
        model.to(device)

        output =  model(coords, aas, mask, auto_regressive=True, temp=temp)

        print(output)

if __name__ == "__main__":
    main()