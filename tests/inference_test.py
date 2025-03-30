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

        model = proteusAI()
        model.to(device)

        output =  model(coords, aas, mask, auto_regressive=True, temp=temp)

        print(output)

if __name__ == "__main__":
    main()