import torch
from tqdm import tqdm
from proteusAI import proteusAI
from utils.train_utils.data_utils import DataHolder

def main():

	device = torch.device("cuda")

	# setup model
	model = proteusAI(old=False, mlm=True, extraction_min_rbf=0.001, extraction_encoder_layers=4)
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/geo_attn_old_4enc_adaptivebias/model_parameters.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/redo_imfuckinglost/model_parameters_e9_s2.26.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/mlm_from_scratch_pure_attn/ctd/model_parameters_e59_s1.74.pth"
	model_path = "/scratch/hjc2538/projects/proteusAI/models/mlm_from_scratch_pure_attn/ctd_maskedchain/model_parameters_e469_s2.21.pth"
	
	model_weights = torch.load(model_path, map_location=device, weights_only=True)
	emb_weights = {".".join(i.split(".")[1:]): model_weights[i] for i in model_weights.keys() if i.startswith("wf_embedding")}
	ext_weights = {".".join(i.split(".")[1:]): model_weights[i] for i in model_weights.keys() if i.startswith("wf_extraction")}
	model.wf_embedding.load_state_dict(emb_weights, strict=True)
	model.wf_extraction.load_state_dict(ext_weights, strict=True)
	model = model.to(device)
	model.eval()

	# setup data
	data = DataHolder(	"/scratch/hjc2538/projects/proteusAI/data/multi_chain/processed", 
						0, 0, -1, # train, val, test samples
						8192, 256, # tokens per batch, max batch size
						64, 8192, # min seq size, max seq size
						True, 3.5, True # use chain mask, max resolution
					)
	data.load("test")

	preds, labels, seq_sims = [],[],[]
	with torch.no_grad():

		pbar = tqdm(total=len(data.test_data), desc="epoch_progress", unit="step")
		for coords_batch, label_batch, chain_idxs, chain_mask, key_padding_mask in data.test_data:
			
			# move to gpu
			label_batch = label_batch.to(device)
			coords_batch = coords_batch.to(device)
			chain_mask = chain_mask.to(device)
			key_padding_mask = key_padding_mask.to(device)
			coords_alpha, coords_beta = model.wf_embedding.get_CaCb_coords(coords_batch, chain_idxs)

			# for old
			# wf = model.wf_embedding(coords_alpha, coords_beta, label_batch, key_padding_mask=key_padding_mask)
			# output = model.wf_extraction(wf, coords_alpha, key_padding_mask=key_padding_mask).argmax(dim=2)
			# seq_sim, seq_pred, seq_true = get_seq_sim(output, label_batch, chain_idxs, key_padding_mask)
			# print(seq_sim)

			# for mlm
			# first test one shot prediction
			aas = -torch.ones_like(label_batch)
			# aas = torch.where(chain_mask, label_batch, -1) # mask token
			# aas = torch.where(chain_mask, -1, -1) # mask token
			# wf = model.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)
			# output = model.wf_extraction(wf, coords_alpha, key_padding_mask=key_padding_mask).argmax(dim=2)
			output = model(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, inference=True)
			# print(output.shape, output)
			seq_sim, seq_pred, seq_true = get_seq_sim(output, label_batch, chain_idxs, key_padding_mask)
			print(seq_sim)

			seq_sims.extend(seq_sim)
			preds.extend(seq_pred)
			labels.extend(seq_true)
			pbar.update(1)

	print(sum(seq_sims)/len(seq_sims))
	with open("results.csv", "w") as f:
		f.write("accuracy,len,chains,pred,true\n")
		for seq_sim, pred, lbl in zip(seq_sims, preds, labels):
			f.write(f"{seq_sim},{len(pred)},{pred.count(":")},{pred},{lbl}\n")

def get_seq_sim(output, labels, chains, mask):

	canonical_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

	seq_sims = []
	seqs = []
	true_seqs = []

	for out, lbl, chain, mask1 in zip(output, labels, chains, mask):
		valid = (~mask1).sum()
		matches = ((out == lbl) & (~mask1)).sum()
		seq_sim = matches / valid

		seq = "".join([canonical_aas[i] for i in out[~mask1]])
		true_seq = "".join([canonical_aas[i] for i in lbl[~mask1]])

		seq_sims.append(seq_sim.item())
		seqs.append(":".join(seq[i:j] for i,j in chain))
		true_seqs.append(":".join(true_seq[i:j] for i,j in chain))

	return seq_sims, seqs, true_seqs


if __name__ == "__main__":
	main()
