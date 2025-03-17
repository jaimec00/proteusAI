import torch
from tqdm import tqdm
from proteusAI import proteusAI
from utils.train_utils.data_utils import DataHolder

def main():

    device = torch.device("cuda")

    # setup model
    model = proteusAI(512,
        False,
        1,
        True, 
        2.0,
        10,
        25,
        2048,
        0,

        8,
        8, 
        True,
        3,
        15,
        1.0,
        32,
        0.001,
        0.85,
        2.0,
        2048,
        0,

        0.1,
        0.0, # attention has less aggressive dropout, as it is already heavily masked
        0.0,

        10.0,
        10.0,

    )
    model_path = "/scratch/hjc2538/projects/proteusAI/models/8enc_8h_32accum/model_parameters.pth"
    # model_path = "/scratch/hjc2538/projects/proteusAI/models/mask/model_parameters.pth"
    model_weights = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(model_weights)
    model = model.to(device)
    model.eval()

    # setup data
    data = DataHolder(	"/scratch/hjc2538/projects/proteusAI/pdb_2021aug02_filtered", 
                        0, 0, -1, # train, val, test samples
                        8192, 256, # tokens per batch, max batch size
                        64, 8192, # min seq size, max seq size
                        True, 3.5 # use chain mask, max resolution
                    )
    data.load("test")

    seq_sims = []
    all_matches = 0
    all_valid = 0

    outputs, labels, coords, chain_idxs_all = [],[],[], []

    with torch.no_grad():

        pbar = tqdm(total=len(data.test_data), desc="epoch_progress", unit="step")

        for label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask in data.test_data:
            
            # move to gpu
            label_batch = label_batch.to(device)
            coords_batch = coords_batch.to(device)
            chain_mask = chain_mask.to(device)
            key_padding_mask = key_padding_mask.to(device)

            # first predict with the true seq as input
            # prediction_batch = torch.nn.functional.one_hot(torch.where(label_batch==-1, 20, label_batch), num_classes=21)
            prediction_batch = torch.nn.functional.one_hot(torch.full_like(label_batch, 20), num_classes=21)


            # run the model
            tot_logits = torch.zeros(label_batch.size(0), label_batch.size(1), 20, device=device)
            for i in range(1):
                # prediction_batch = torch.nn.functional.one_hot(torch.randint_like(label_batch, 0,21), num_classes=21)
                # noise = torch.randn_like(coords_batch)*0.1

                output = model(coords_batch, prediction_batch, chain_idxs, key_padding_mask, mask_predict=False, temp=1e-6, num_iters=5, remask=True)
                tot_logits += output
		
            tot_logits = (tot_logits/1)
            output = torch.softmax(tot_logits/1e-6, dim=2)

            # compute seq sims
            seq_sim, matches, valid = get_seq_sim(output, label_batch, key_padding_mask)
            seq_sims.append(seq_sim)
            all_matches += matches
            all_valid += valid
            outputs.append(output)
            labels.append(label_batch)
            coords.append(coords_batch)
            chain_idxs_all.append(chain_idxs)
            pbar.update(1)

    
    seq_sims = torch.tensor(seq_sims)

    print(seq_sims, seq_sims.mean(), seq_sims.min(), seq_sims.max(), seq_sims.std())
    print(all_matches/all_valid)
    best = torch.argmax(seq_sims)
    best_sample = outputs[best].argmax(dim=2)
    best_labels = labels[best]
    best_chains = chain_idxs_all[best]

    best_prot = ((best_sample == best_labels).sum(dim=1) / (best_labels!=-1).sum(dim=1))

    print(best_prot)

    best_pred = best_sample[best_prot.argmax(dim=0)]
    best_prot_lbl = best_labels[best_prot.argmax(dim=0)]
    best_chains = best_chains[best_prot.argmax(dim=0)]

    canonical_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    seq = "".join([canonical_aas[i] for i in best_pred[best_prot_lbl!=-1]])
    true_seq = "".join([canonical_aas[i] for i in best_prot_lbl[best_prot_lbl!=-1]])
    print(best_chains, len(seq))
    print(seq, true_seq)

def get_seq_sim(output, labels, mask):
    mask_flat = mask.view(-1)
    labels_flat = labels.view(-1)[~mask_flat]
    output_flat = output.argmax(dim=2).view(-1)[~mask_flat]

    matches = torch.sum(output_flat == labels_flat) 
    valid = (~mask_flat).sum()

    seq_sim = matches / valid

    return seq_sim, matches, valid

if __name__ == "__main__":
    main()