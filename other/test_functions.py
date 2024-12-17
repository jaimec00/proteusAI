from collections import defaultdict
import plotly.graph_objects as go
from Bio.PDB import PDBParser
from pathlib import Path
from tqdm import tqdm
import torch
import math
import cv2
import sys
import os


from utils.data_utils import pdb_to_torch
from utils.model_utils import protein_to_wavefunc
from proteusAI import proteusAI

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def main():

	# test_wf(Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02_filtered_ncaa/pdb/lw/1lwu_A_B_C_J_K_L_M_P.pt"))
	# test_model()
	test_sparse_MHA()

def test_sparse_MHA():
	batch, N, nhead, d_model = 4, 10, 4, 32
	d_k = d_model // nhead

	q = torch.randn(batch, N, d_model)
	k = torch.randn(batch, N, d_model)
	v = torch.randn(batch, N, d_model)

	mask = torch.zeros(batch, N, )
	mask[:, N//2:] = True # mask last half of each batch
	
	q_proj = torch.randn(batch, nhead, N, d_k)
	k_proj = torch.randn(batch, nhead, N, d_k)
	v_proj = torch.randn(batch, nhead, N, d_k)

	q = 

	pass

def test_model():

	batch = 4
	N = 10000
	d_model = 128

	gpu = torch.device("cuda")

	model = proteusAI(
			N=N, 
			d_model=d_model, 
			n_head=4, 
			decoder_layers=4, 
			hidden_linear_dim=d_model*2, 
			dropout=0.0, 
			active_decoders=-1, 
			use_probs=False, 
	).to(gpu)

	coords = torch.rand(batch, N, 3).to(gpu)
	features = torch.rand(batch, N, d_model).to(gpu)
	predictions = torch.zeros(batch, N, 21, dtype=torch.long).to(gpu)
	mask = torch.zeros(batch, N, dtype=torch.bool).to(gpu)

	print("coords: ", coords, coords.shape, sep="\n")
	print("features: ", features, features.shape, sep="\n")
	print("predictions: ", predictions, predictions.shape, sep="\n")
	print("mask: ", mask, mask.shape, sep="\n")

	output = model(coords, predictions, features=features, key_padding_mask=mask, auto_regressive=False, temp=0.1, use_checkpoint=False)

	print(f"output: \n{output}\n{output.shape}")

	amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
	aa_decoder = {idx: aa for idx, aa in enumerate(amino_acids)}
	aas = output.argmax(dim=-1) # batch x N
	for idx, batch in enumerate(aas):
		sequence = "".join(aa_decoder[aa] for aa in batch)
		print(f"sequence {idx}: {sequence}")


def multi_gpu_test():

	dist.init_process_group(backend='nccl')
	local_rank = dist.get_rank()
	device = torch.device(f"cuda:{local_rank}")

	if local_rank==0:
		x = torch.tensor([  [1,2],
							[3,4]  ])
	else:
		x = None

	x = dist.broadcast_object_list([x], src=0)[0]
	x = torch.tensor(x, device=device)

	chunks = x.chunk(2, dim=0)

	chunk = chunks[local_rank].to(device)

	out_chunk = chunk*2

	# Gather the results back to rank 0
	gathered_output = [torch.zeros_like(out_chunk) for _ in range(dist.get_world_size())]
	dist.all_gather(gathered_output, out_chunk)

	# Concatenate and print output on rank 0
	if local_rank == 0:
		output = torch.cat(gathered_output, dim=0)
		print(f"Final output: \n{output}")

	# with torch.cuda.stream()


def inference(max_tokens=512, d_model=512):

	print("loading model weights...")
	base = Path("/gpfs_backup/wangyy_data/protAI")
	model_path = base / Path("models/run4/model_parameters.pth")
	state_dict = torch.load(model_path, weights_only=True)
	
	N, d_model, n_head, encoder_layers, decoder_layers, hidden_linear_dim, dropout, use_features = 512, 512, 8, 4, 4, 1024, 0.1, True
	model = proteusAI(N, d_model, n_head, encoder_layers, decoder_layers, hidden_linear_dim, dropout, use_features)
	model.load_state_dict(state_dict)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	test = base / Path("pmpnn_data/pdb_2021aug02/pdb/lw/_wf.pt")

	features_and_labels = torch.load(test, weights_only=True)
	unpadded_features = features_and_labels["features"].to(device)
	unpadded_labels = features_and_labels["labels"].to(device)

	features = torch.cat([unpadded_features, torch.zeros(max_tokens - unpadded_features.size(0), d_model, device=device)], dim=0)
	labels = torch.cat([unpadded_labels, -torch.ones(max_tokens - unpadded_labels.size(0), device=device)], dim=0)

	features = features.unsqueeze(0)
	labels = labels.unsqueeze(0)
	key_padding_mask = labels == -1
	prediction = torch.full([labels.size(0), labels.size(1), 20], 1/20, device=device)

	model.eval()
	print("starting inference")

	with torch.no_grad():
		# one-hot
		output = model(features, prediction, key_padding_mask, diffusion_cycles = 5, auto_regressive=True, temp=0.01)
	
	output = torch.argmax(output, dim=-1)
	aas = "ACDEFGHIKLMNPQRSTVWY"
	sequence = "".join(aas[i] for i in output[0, ~key_padding_mask.squeeze(0)])

	similarity = 100 * torch.sum(labels[~key_padding_mask] == output[~key_padding_mask]) / torch.sum(~key_padding_mask, dim=1)

	print(similarity, torch.sum(~key_padding_mask), sequence)

def test_noising():

	batch, N, features = [2, 3, 3]

	label_batch = torch.tensor([  
									[0, 1, 2],
									[0, 1, -1]
								])

	key_padding_mask = label_batch != -1

	print(label_batch)
	print(key_padding_mask)

	noised_labels = smooth_and_noise_labels(label_batch, 1/5,0,0.1, num_classes=3)

	print()
	print(noised_labels)

	mean_cel, mean_seq_sim = compute_cel_and_seq_sim(label_batch, noised_labels, features)

	print(mean_cel, mean_seq_sim)

def test_wf(pdb=Path("1rvz.pdb")):

	with torch.no_grad():

		d_model = 128
		min_wl = 3.7
		max_wl = 20
		base = 30

		if pdb.name.endswith(".pdb"):

			# load coords and expand to get batch x N x 3
			parser = PDBParser(QUIET=True)
			coords = pdb_to_torch(pdb, parser)


			coords_batch = coords.unsqueeze(0) # 1 x N x 3 

			# create dummy key padding mask ; batch X N
			key_padding_mask = torch.zeros(1, coords.size(0), dtype=torch.bool)

			# get features and wavelengths
			features, wavelengths = protein_to_wavefunc(coords_batch, key_padding_mask, d_model=d_model, return_wl=True, min_wl=min_wl, max_wl=max_wl, base=base, device="cpu")
		
			# squeeze to remove batch dimension
			features = features.squeeze(0)
			wavelengths = wavelengths.squeeze(0)
		
		elif pdb.name.endswith(".pt"):
			torch.serialization.add_safe_globals([defaultdict, list])
			pdb_info = torch.load(pdb, weights_only=True)

			coords = pdb_info["coords"]
			features = pdb_info["features"]

			log_distribution = (torch.logspace(0, 1, d_model//2, base=base) - 1) / (base - 1) # Scale [1, 2) to [0, 1)
			wavelengths = (min_wl + (log_distribution.mul_(max_wl - min_wl))) # num_wl,

		# make out dir
		outdir = Path(f"wf_plots/test{d_model}_{base}_{pdb.name.rstrip('.pdb')}")
		outdir.mkdir(parents=True, exist_ok=True)

		# plot each wave function
		for i, j, wavelength in tqdm(zip(range(0, d_model, 2), range(1, d_model, 2), wavelengths)):
			real = features[:, i]
			imag = features[:, j]

			# print(torch.max(real), torch.min(real), torch.max(imag), torch.min(imag))

			imag_norm = (imag - imag.min()) / (imag.max() - imag.min())

			real_norm = (((real - real.min()) / (real.max() - real.min())) * 5) + 1

			# Apply the function to convert each normalized value to a hex color
			hex_colors = [gradient_color(val.item()) for val in imag_norm]

			plot_tensor(coords, wavelength, hex_colors, real_norm, save=outdir)

		# create video
		for frame_duration in [0.15, 0.35, 0.5]:
			create_video_from_images(outdir, f"{outdir}/{outdir.name.split('.')[0]}_{frame_duration}.mp4", frame_duration)

# Function to map normalized values (0 to 1) to a gradient from red to blue
def gradient_color(val):
	# val is between 0 and 1; for red to blue gradient:
	r = int(255 * (1 - val))  # Red decreases as val increases
	g = 0                     # Green stays zero
	b = int(255 * val)         # Blue increases as val increases
	return f'#{r:02x}{g:02x}{b:02x}'  # Convert to hex

def plot_tensor(coords, wl=0, color=None, marker_size=None, save=None):

	# Use Plotly to plot the 3D scatter plot directly
	fig = go.Figure(data=[go.Scatter3d(
		x=coords[:, 0].numpy(),
		y=coords[:, 1].numpy(),
		z=coords[:, 2].numpy(),
		mode='markers',
		marker=dict(
			size=marker_size.numpy() if marker_size is not None else 10, # magnitude based
			color=color,  # Color by phase
			colorscale='HSV',  # make color scale cyclic
			opacity=0.8
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
		title=f"Lambda: {round(float(wl),5)} A",
		title_x=0.5  # Centers the title
	)

	# Show the plot
	if save:
		fig.write_image(f"{save}/lambda_{round(float(wl), 5)}.png")
	else:
		fig.show()

def create_video_from_images(image_folder, output_video, frame_duration):
	# Get the list of image files from the folder (assuming PNGs)
	images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	
	# Sort the images based on filename to ensure correct order
	images = sorted(images, key=lambda x: float(x.split("_")[1].rstrip(".png")))

	# Load the first image to get the dimensions (height, width)
	first_image_path = os.path.join(image_folder, images[0])
	frame = cv2.imread(first_image_path)
	height, width, layers = frame.shape

	# Set the frames per second (fps) based on the frame duration (in seconds)
	fps = 1.0 / frame_duration

	# Initialize the video writer object
	video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

	# Add each image to the video
	for image in images:
		image_path = os.path.join(image_folder, image)
		frame = cv2.imread(image_path)
		video_writer.write(frame)

	# Release the video writer object
	video_writer.release()

	print(f"Video saved as {output_video}")

if __name__ == "__main__":
	main()