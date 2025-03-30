import torch
from tqdm import tqdm
from proteusAI import proteusAI
from utils.train_utils.data_utils import DataHolder
from utils.model_utils.base_modules.base_modules import CrossFeatureNorm
def main():

    device = torch.device("cuda")

    # setup model
    model = proteusAI(
                        d_model=512, num_aas=20, 

						# wf embedding params
						embedding_min_wl=2, embedding_max_wl=10, embedding_base_wl=25, embedding_learnable_aa=False,

						# wf diffusion params
						diffusion_beta_min=1e-4, diffusion_beta_max=0.02, diffusion_beta_schedule_type="linear", diffusion_t_max=100,
						diffusion_min_wl=0.001, diffusion_max_wl=100, # for sinusoidal timestep embedding
						diffusion_mlp_timestep=False, diffusion_d_hidden_timestep=2048, diffusion_hidden_layers_timestep=0, diffusion_norm_timestep=True,
						diffusion_mlp_pre=True, diffusion_d_hidden_pre=2048, diffusion_hidden_layers_pre=0, diffusion_norm_pre=True,
						diffusion_mlp_post=True, diffusion_d_hidden_post=2048, diffusion_hidden_layers_post=0, diffusion_norm_post=True,
						diffusion_encoder_layers=8, diffusion_heads=8, diffusion_learnable_spreads=True,
						diffusion_min_spread=1.5, diffusion_max_spread=8.0, diffusion_base_spreads=1.0, diffusion_num_spread=32,
						diffusion_min_rbf=0.001, diffusion_max_rbf=0.85, diffusion_beta=2.0,
						diffusion_d_hidden_attn=2048, diffusion_hidden_layers_attn=0,

						# wf extraction params
						extraction_mlp_pre=True, extraction_d_hidden_pre=2048, extraction_hidden_layers_pre=0, extraction_norm_pre=True,
						extraction_mlp_post=True, extraction_d_hidden_post=2048, extraction_hidden_layers_post=0, extraction_norm_post=True,
						extraction_encoder_layers=8, extraction_heads=8, extraction_learnable_spreads=True,
						extraction_min_spread=3.0, extraction_max_spread=15.0, extraction_base_spreads=1.0, extraction_num_spread=32,
						extraction_min_rbf=0.001, extraction_max_rbf=0.85, extraction_beta=2.0,
						extraction_d_hidden_attn=2048, extraction_hidden_layers_attn=0,

						# dropout params
						dropout=0.10, attn_dropout=0.00, wf_dropout=0.00,
    )
    model_path = "/scratch/hjc2538/projects/proteusAI/models/diffusion_debugged/model_parameters_e9_s0.0.pth"
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

    norm = CrossFeatureNorm(d_model=512)

    with torch.no_grad():

        pbar = tqdm(total=len(data.test_data), desc="epoch_progress", unit="step")

        for coords_batch, label_batch, chain_idxs, chain_mask, key_padding_mask in data.test_data:
            
            # move to gpu
            label_batch = label_batch.to(device)
            coords_batch = coords_batch.to(device)
            chain_mask = chain_mask.to(device)
            key_padding_mask = key_padding_mask.to(device)

            coords_alpha, coords_beta = model.get_CaCb_coords(coords_batch, chain_idxs)

            # first predict with the true seq as input
            # prediction_batch = torch.nn.functional.one_hot(torch.where(label_batch==-1, 20, label_batch), num_classes=21)
            # aas = label_batch # first see if can actually denoise and reconstruct seq
            aas = -torch.ones_like(label_batch)

            true = model.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask)

            # t = 100
            t = torch.full((1,1,1), 100)

            
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

            # run the model
            output = model(coords_batch, aas=aas, chain_idxs=chain_idxs, key_padding_mask=key_padding_mask,
                            inference=True,
                            cycles=100, 
                            temp=1e-6
                            )

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

            break

    
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