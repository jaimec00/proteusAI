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

def main():

	test_wf()



def test_wf(pdb=Path("1rvz.pdb")):

	with torch.no_grad():

		d_model = 128
		min_wl = 3.7
		max_wl = 20
		base = 20

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