import torch
from tqdm import tqdm
from proteusAI import proteusAI
from utils.train_utils.data_utils import DataHolder
from utils.model_utils.base_modules.base_modules import CrossFeatureNorm
def main():

    device = torch.device("cuda")

    # setup model
    model = proteusAI(diffusion_encoder_layers=1, diffusion_use_bias=True)
    model_path = "/scratch/hjc2538/projects/proteusAI/models/DIFFUSION_biasedattn_noadaln/model_parameters.pth"
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

        for coords_batch, label_batch, chain_idxs, chain_mask, key_padding_mask in data.test_data:
            
            # move to gpu
            label_batch = label_batch.to(device)
            coords_batch = coords_batch.to(device)
            chain_mask = chain_mask.to(device)
            key_padding_mask = key_padding_mask.to(device)

            coords_alpha, coords_beta = model.wf_embedding.get_CaCb_coords(coords_batch, chain_idxs)

            # first predict with the true seq as input
            # prediction_batch = torch.nn.functional.one_hot(torch.where(label_batch==-1, 20, label_batch), num_classes=21)
            aas = label_batch # first see if can actually denoise and reconstruct seq
            # aas = -torch.ones_like(label_batch)
            # aas = torch.full_like(label_batch, 5)

            # wf = model.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask)
            wf_no_aa = model.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask, no_aa=True)
            wf = wf_no_aa

            latent_wf = model.wf_encoding.encode(wf, coords_alpha, key_padding_mask=key_padding_mask, wf_no_aa=wf_no_aa)
            print(latent_wf)

            decode_wf_nonoise = model.wf_decoding(latent_wf, coords_alpha, key_padding_mask=key_padding_mask, wf_no_aa=wf_no_aa)
            output_nonoise = model.wf_extraction.extract(decode_wf_nonoise, coords_alpha, key_padding_mask, wf_no_aa=wf_no_aa)
            seq_sim_nonoise, matches, valid = get_seq_sim(output_nonoise, label_batch, key_padding_mask)
            
            t = 5

            t_tensor = torch.tensor([t], device=wf.device).unsqueeze(0).unsqueeze(1).expand(wf.size(0), 1, 1)
            abar, _ = model.wf_diffusion.noise_scheduler(t_tensor)

            latent_noise, noise = model.wf_diffusion.noise(latent_wf, t)
            noise_pred = model.wf_diffusion(latent_noise, coords_alpha, t_tensor, key_padding_mask=key_padding_mask)
            # print((noise_pred-noise)**2)
            latent_wf = model.wf_diffusion.denoise(latent_noise, coords_alpha, t, key_padding_mask=key_padding_mask, wf_no_aa=wf_no_aa)

            decode_wf = model.wf_decoding(latent_wf, coords_alpha, key_padding_mask=key_padding_mask, wf_no_aa=wf_no_aa)
            output = model.wf_extraction.extract(decode_wf, coords_alpha, key_padding_mask, wf_no_aa=wf_no_aa)

            wf = model.wf_embedding(coords_alpha, coords_beta, output, key_padding_mask)
            latent_wf = model.wf_encoding.encode(wf, coords_alpha, key_padding_mask=key_padding_mask, wf_no_aa=wf_no_aa)

            seq_sim, matches, valid = get_seq_sim(output, label_batch, key_padding_mask)

            print(seq_sim, seq_sim_nonoise, abar.mean().item())




            # noised_wf, noise = model.wf_diffusion.noise(true, t=t)

            # pred_wf = model.wf_diffusion.denoise(noised_wf, coords_alpha, t_start=t, key_padding_mask=key_padding_mask)
            # pred_wf = pred_wf.masked_fill(key_padding_mask.unsqueeze(2), 0)
            # true = true.masked_fill(key_padding_mask.unsqueeze(2), 0)
            # # pred_wf = norm(pred_wf, key_padding_mask)

            # loss = (true-pred_wf)**2
            # baseloss = (true-noised_wf)**2

            # # print(pred_wf, true)

            # print(loss.sum() / (512*(~key_padding_mask).sum()))
            # print(baseloss.sum() / (512*(~key_padding_mask).sum()))

            # # print(loss.sqrt() / true)

            # output = model.wf_extraction.extract(pred_wf, coords_alpha, key_padding_mask, temp=1e-6)

            # # run the model
            # output = model(coords_alpha=coords_alpha, coords_beta=coords_beta, aas=aas, key_padding_mask=key_padding_mask,
            #                 inference=True, t=t,
            #                 cycles=1, 
            #                 temp=1e-6
            #                 )

            # compute seq sims

            # seq_sims.append(seq_sim)
            # all_matches += matches
            # all_valid += valid
            # outputs.append(output)
            # labels.append(label_batch)
            # coords.append(coords_batch)
            # chain_idxs_all.append(chain_idxs)
            # pbar.update(1)

            # break

    
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
    output_flat = output.view(-1)[~mask_flat]

    matches = torch.sum(output_flat == labels_flat) 
    valid = (~mask_flat).sum()

    seq_sim = matches / valid

    return seq_sim, matches, valid

if __name__ == "__main__":
    main()