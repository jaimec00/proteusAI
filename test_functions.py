import torch
import plotly.graph_objects as go
from get_data import pdb_to_torch
from Bio.PDB import PDBParser
import sys
from pathlib import Path
import math
import cv2
import os
from utils import protein_to_wavefunc
from tqdm import tqdm
import torch
from utils import smooth_and_noise_labels, compute_cel_and_seq_sim

def main():
    test_wf()

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

def test_wf():
    # load coords and expand to get batch x N x 3
    parser = PDBParser(QUIET=True)
    coords = pdb_to_torch(Path("test_pdbs/clean_pdbs/1rvz.pdb"), parser)
    coords_batch = coords.expand(1,-1,-1)

    # create dummy key padding mask ; batch X N
    key_padding_mask = torch.zeros([1, coords.size(0)], dtype=torch.bool)

    # get features and wavelengths
    d_model = 512
    features, wavelengths = protein_to_wavefunc(coords_batch, key_padding_mask, d_model=d_model, return_wl=True)
    
    # squeeze to remove batch dimension
    features = features.squeeze(0)
    wavelengths = wavelengths.squeeze(0)
    
    # make out dir
    outdir = Path("wf_plots/1rvz_logScale_20_2")
    outdir.mkdir(parents=True, exist_ok=True)

    # plot each wave function
    for i, j, wavelength in tqdm(zip(range(0, d_model, 2), range(1, d_model, 2), wavelengths)):
        real = features[:, i]
        imag = features[:, j]


        imag_norm = (imag - imag.min()) / (imag.max() - imag.min())

        real_norm = (((real - real.min()) / (real.max() - real.min())) * 25) + 1

        # Apply the function to convert each normalized value to a hex color
        hex_colors = [gradient_color(val.item()) for val in imag_norm]

        plot_tensor(coords, wavelength, hex_colors, real_norm, save=outdir)

    # create video
    frame_duration = 0.5
    create_video_from_images(outdir, f"{outdir}/{outdir.name}_{frame_duration}.mp4", frame_duration)

# Function to map normalized values (0 to 1) to a gradient from red to blue
def gradient_color(val):
    # val is between 0 and 1; for red to blue gradient:
    r = int(255 * (1 - val))  # Red decreases as val increases
    g = 0                     # Green stays zero
    b = int(255 * val)         # Blue increases as val increases
    return f'#{r:02x}{g:02x}{b:02x}'  # Convert to hex

def plot_tensor(coords, wl, color=None, marker_size=None, save=None):

    # Use Plotly to plot the 3D scatter plot directly
    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0].numpy(),
        y=coords[:, 1].numpy(),
        z=coords[:, 2].numpy(),
        mode='markers',
        marker=dict(
            size=marker_size.numpy() if marker_size is not None else None, # magnitude based
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