import torch
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm
import os
from proteusAI import proteusAI
from sklearn.decomposition import PCA

# for loading the biounits
from collections import defaultdict
torch.serialization.add_safe_globals([defaultdict, list])

def main():

	# define device
	device = "cuda"

	# load an example pdb
	example_pt = Path("/scratch/hjc2538/projects/proteusAI/pdb_2021aug02_filtered/pdb/rv/1rvz_1.pt")
	example_pdb = torch.load(example_pt, weights_only=True)
	
	# define output path
	base_dir = Path(os.path.abspath(__file__)).parent.parent # this file is in the test dir, base is one dir above
	out_path = base_dir / Path("plots") / Path(example_pt.name.split(".")[0])
	out_path.mkdir(parents=True, exist_ok=True) # make the output dir

	# load the data of the example pdb
	# also expand so batch dim included, chains is wrapped in a list to simulate this
	# first do pca on coords so that plotting is cleaner
	coords = example_pdb["coords"]
	coords = coords - coords.mean(dim=1, keepdim=True)
	pca = PCA(n_components=3)
	pca.fit(coords)
	coords = pca.transform(coords)
	coords = torch.tensor(coords[None, :, :]).to(torch.float32).to(device)

	labels = example_pdb["labels"][None, :].to(device)
	mask = labels==-1 
	chains = [list(example_pdb["chain_idxs"].values())]
	
	# define the pretrained weights
	# embedding_weights =  Path("/scratch/hjc2538/projects/proteusAI/models/extraction_4_learnwavenumbers/tmp/model_parameters_embedding.pth")
	# extraction_weights = Path("/scratch/hjc2538/projects/proteusAI/models/extraction_4_learnwavenumbers/tmp/model_parameters_extraction.pth")
	
	embedding_weights = Path("/scratch/hjc2538/projects/proteusAI/models/vanilla_model_12enc/model_parameters_embedding.pth")
	extraction_weights = Path("/scratch/hjc2538/projects/proteusAI/models/vanilla_model_12enc/model_parameters_extraction.pth")
	
	# init the model and load the weights
	model = proteusAI(old=True, extraction_encoder_layers=12, extraction_hidden_layers_pre=2, extraction_heads=16) # assume the model defaults align with the weights, only ones that matter for this test are embedding and extraction though
	model.load_WFEmbedding_weights(embedding_weights, device)
	model.load_WFExtraction_weights(extraction_weights, device)
	model.to(device)

	model.eval() # make sure in evaluation mode
	with torch.no_grad():
		
		# get alpha and beta carbons
		Ca, Cb = model.wf_embedding.get_CaCb_coords(coords, chains)

		# compute wf
		wf = model.wf_embedding(Ca, Cb, labels, key_padding_mask=mask)

		# preprocess the wf, do it manually instead of calling extraction forward to keep intermediates
		wf = model.wf_extraction.norm_pre(wf + model.wf_extraction.mlp_pre(wf))

		# for now will only work with a single encoder, the first is most likely to capture the spatial relationships, i think
		# project the tensors
		Q = torch.matmul(wf.unsqueeze(1), model.wf_extraction.encoders[0].attn.q_proj.unsqueeze(0)) + model.wf_extraction.encoders[0].attn.q_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		K = torch.matmul(wf.unsqueeze(1), model.wf_extraction.encoders[0].attn.k_proj.unsqueeze(0)) + model.wf_extraction.encoders[0].attn.k_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		V = torch.matmul(wf.unsqueeze(1), model.wf_extraction.encoders[0].attn.v_proj.unsqueeze(0)) + model.wf_extraction.encoders[0].attn.v_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		
		# compute attention matrix
		attn_logits = torch.matmul(Q, K.transpose(2,3)) / (Q.size(3)**0.5)
		attn = attn_logits.masked_fill(mask[:, None, :, None] | mask[:, None, None, :], float("inf")).squeeze(0) # H x N x N
		attn = attn - attn.min(dim=2, keepdim=True).values
		attn = attn / (attn.max(dim=2, keepdim=True).values) # scale so max val is 1

	# now that have attention weights, want to plot them onto the protein coords
	# plan is to loop through the residues, color that one green, then do a heatmap, where blue mean 0 and red mean 1
	for head_idx, head in enumerate(attn): # head is N x N now

		# make a directory for each head
		head_outdir = out_path / Path(f"head_{head_idx}")
		head_outdir.mkdir(exist_ok=True) 

		# loop through the residues
		for resi_idx, resi in tqdm(enumerate(head), desc=f"plotting head {head_idx}", total=head.size(0)): # resi is N now
			
			# define the filename for each resi
			resi_plot_path = head_outdir / Path(f"resi_{resi_idx}.png")

			# values already normalized by softmax, just need to convert to hex
			attn_colors = [gradient_color(val.item()) if resi_idx!=idx else '#00ff00' for idx, val in enumerate(resi)] # resi we plot attn for is green

			# plot the attention heatmap
			plot_attn(Ca[0, :, :].cpu(), color=attn_colors, save=resi_plot_path, head_idx=head_idx, resi_idx=resi_idx)


# Function to map normalized values (0 to 1) to a gradient from red to blue
def gradient_color(val):
	# val is between 0 and 1; for red to blue gradient:
	r = int(255 * val)        # Red increases as val increases
	g = 0                     # Green stays zero
	b = int(255 * (1 - val))  # Blue decreases as val increases
	return f'#{r:02x}{g:02x}{b:02x}'  # Convert to hex

def plot_attn(coords, color=None, marker_size=None, save=None, head_idx=0, resi_idx=0):

	# Use Plotly to plot the 3D scatter plot directly
	fig = go.Figure(data=[go.Scatter3d(
		x=coords[:, 0].numpy(),
		y=coords[:, 1].numpy(),
		z=coords[:, 2].numpy(),
		mode='markers',
		marker=dict(
			size=marker_size.numpy() if marker_size is not None else 4, # magnitude based
			color=color,  # Color by phase
			colorscale='HSV',  # make color scale cyclic
			opacity=0.25
		)
	)])

	# Set titles and axis labels
	fig.update_layout(
		scene=dict(
			xaxis_title='X Coordinate',
			yaxis_title='Y Coordinate',
			zaxis_title='Z Coordinate'
		)
	)

	# Add a title to the plot
	fig.update_layout(
		title=f"Attention Heatmap for Head {head_idx} and Residue {resi_idx}",
		title_x=0.5  # Centers the title
	)

	# Show the plot
	fig.write_image(save)

if __name__ == "__main__":
	main()