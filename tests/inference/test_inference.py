import torch
from tqdm import tqdm
from proteusAI import proteusAI
from utils.train_utils.data_utils import DataHolder
# from utils.inference_utils.inference_utils import load_model
from data.constants import canonical_aas, alphabet

def main():

	device = torch.device("cuda")

	# setup model
	# model_path = "/storage/cms/wangyy_lab/hjc2538/proteusAI/models/multichain/ca/32seqsim/model_parameters_e99_s2.16.pth"	
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/big_test/model_parameters_e69_s2.19.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/geo_attn_2kaccum_4h_4enc_256dim_weightdec1pct_001mask_drop10/model_parameters_e119_s2.21.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/recycles_d256/model_parameters_e79_s2.2.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/norecycle_12enc_d256/model_parameters_e119_s2.15.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/redo_imfuckinglost/model_parameters_e9_s2.26.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/mlm_from_scratch_pure_attn/ctd/model_parameters_e59_s1.74.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/mlm_from_scratch_pure_attn/ctd_maskedchain/model_parameters_e469_s2.21.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/mlm_selfupdate/model_parameters_e99_s2.2.pth"
	# model_path = "/scratch/hjc2538/projects/proteusAI/models/gnn/attn_reduction/checkpoint_e109_s1.44.pth"
	model_path = "/scratch/hjc2538/projects/proteusAI/models/gnn/no_decoder/checkpoint_e69_s1.5.pth"

	model_weights = torch.load(model_path, map_location=device, weights_only=True)["model"]
	# model = load_model(model_weights)
	model = proteusAI(enc_layers=6, dec_layers=0)
	model.load_state_dict(model_weights)
	model = model.to(device)
	model.eval()

	# setup data
	data = DataHolder(	"/scratch/hjc2538/projects/proteusAI/data/multi_chain/processed", 
						0, 0, -1, # train, val, test samples
						8192, 256, # tokens per batch, max batch size
						64, 8192, # min seq size, max seq size
						True, 3.5, False # use chain mask, max resolution, ca only
					)
	data.load("test")

	preds, labels, seq_sims = [],[],[]
	with torch.no_grad():

		pbar = tqdm(total=len(data.test_data), desc="epoch_progress", unit="step")
		for data_batch in data.test_data:
			
			# move to gpu
			label_batch = data_batch.labels.to(device)
			coords_batch = data_batch.coords.to(device)
			chain_mask = data_batch.chain_masks.to(device) | data_batch.homo_masks.to(device)
			chain_idxs = data_batch.chain_idxs
			key_padding_mask = data_batch.key_padding_masks.to(device)

			# first test one shot prediction
			aas = torch.where(chain_mask, -1, label_batch)

			# this is how pmpnn does the decoding order so that visible chains more likely to be predicted first. bit obtuse but whatever
			rand_vals = (chain_mask+0.0001) * torch.abs(torch.randn_like(chain_mask, dtype=torch.float32))
			rand_vals = torch.where(key_padding_mask, float("inf"), rand_vals) # masked positions decoded last to make inference quicker
			decoding_order = torch.argsort(rand_vals, dim=1) # Z x N

			output = model(coords_batch, aas, chain_idxs, key_padding_mask, decoding_order, inference=False, temp=1e-6).argmax(2)
			seq_sim, seq_pred, seq_true = get_seq_sim(output, label_batch, chain_idxs, key_padding_mask, ~chain_mask)

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

def get_seq_sim(output, labels, chains, mask, chain_mask):

	seq_sims = []
	seqs = []
	true_seqs = []

	for out, lbl, chain, mask1, chain_mask1 in zip(output, labels, chains, mask, chain_mask):
		all_mask = ~(mask1 | chain_mask1)
		valid = (all_mask).sum()
		matches = ((out == lbl) & all_mask).sum()
		seq_sim = matches / valid

		seq = "".join([canonical_aas[i] for i in out[~mask1]])
		true_seq = "".join([alphabet[i] for i in lbl[~mask1]])

		seq_sims.append(seq_sim.item())
		seqs.append(":".join(seq[i:j] for i,j in chain))
		true_seqs.append(":".join(true_seq[i:j] for i,j in chain))

	return seq_sims, seqs, true_seqs


if __name__ == "__main__":
	main()
