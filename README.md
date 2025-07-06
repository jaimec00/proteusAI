# proteusAI
## Protein Sequence Prediction of Target Structure via Wave Function Embedding, Message Passing, and Masked Language Modeling

## Summary

proteusAI is masked language model for protein sequence prediction conditioned on structure. It is based on ProteinMPNN, but with a few innovations and customizations aimed to better suit the inverse folding problem.

The key innovation is the use of wavefunction embedding to initialize the nodes in the original graph. ProteinMPNN initializes the nodes as zeros, and allows the edges to update them. We aimed to give the model
a global view of the protein by encoding the environment of each residue through the superposition of the effects of all other residues. We accomplish this by modeling each $C_\alpha$ as an anisotropic point source, where the anisotropy comes from the orientation of the virtual $C_\beta$. We evaluate multiple wavefunction, each using a learnable wavenumber that corresponds to a feature index. Additionally we chose to encode the identity of each amino acid by scaling the magnitude of the corresponding $C_\beta$, with a learnable scaling factor per wavenumber, per amino acid. This allows the model to learn how each amino acid affects the environment around it, at each scale, and most importantly, in the context of its spatial configuration. once the embedding is computed, we norm the features and pass it through a linear layer. 

The edge features are computed the same way as ProteinMPNN does, through 16 radial basis functions with evenly spaced centers from 2 to 22 $\AA$. The edge features are also normed and passed through a linear layer.

After embedding the global environment of each residue along with the nearest-neighbor relationships, we pass it through several encoder layers, which are identical to those used by ProteinMPNN, in which several message passing operations are executed, each of which updates the node and edge representations. We ommited the decoder implemented by ProteinMPNN, instead treating the encoder as a bidirectional encoder that predicts the amino acid identity of positions with the special \<mask\> token. 

For training, we allowed the model to start with the true amino acids for non-target chains, but set all of the amino acids for the target chain to the special \<mask\> token. We used an alphafold style recycle regimen, where we randomly sampled a number between 0 and $N_\text{recycle}$-1, ran the model with no gradients for that number of cycles, and randomly replaced 1/$N_\text{recycle}$ of residues with the models predictions. this served as input to the model, with gradients on. This allows the model to be trained not only with sequences in the PDB as context, but also with viable alternative sequences, as predicted by the model, in the hope that it will learn more robust structure to sequence relationships not captured by evolution.

### Getting Started
#### Installation
First you will need a working Linux machine with a GPU set up on it, preferably with an H100 GPU, as the wave function embedding CUDA kernel was optimized for this hardware (mostly the generous amount of shared memory). 

Assuming this, you will run the setup.sh script, which creates a conda environment named protAI_env:

```bash
./setup.sh
```
We are currently working on a script to automatically compile the cuda kernel, but for now you can do it manually

```python
conda activate protAI_env

# go to path for learnable wave function embedding 
cd utils.model_utils/wf_embedding/anisotropic/aa_scaling/learnable_aa/learnable_wavenumber/cuda
python setup.py build_ext --inplace

# go to path for static wave function embedding (much faster)
cd utils.model_utils/wf_embedding/anisotropic/aa_scaling/static_aa/cuda
python setup.py build_ext --inplace

```

This will compile the kernels so they can be used in training and inference.

#### Inference
TBD

#### Training
##### Data Preperation
We use two data sets to train this model. 

    Multi-chain: https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz 

    Single-chain: https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/{chain_set.jsonl,chain_set_splits.json} 

You can download the data sets like this

```bash
# download multi chain
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz /path/to/multichain/data
nohup tar -xzv /path/to/multchain/data/pdb_2021aug02.tar.gz & # recommended to run in background with nohup, takes a while

# download single chain
wget  https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl /path/to/singlechain/data
wget  https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json /path/to/singlechain/data
```

You will then have to clean the data so that it is compatible with the training script. You can do this easily by running the following commands from the proteusAI directory

```bash
# clean multi-chain data
nohup python utils/train_utils/data_utils   --clean_pdbs True \
                                            --data_path '/path/to/multichain/data/pdb_2021aug02'\
                                            --new_data_path '/path/to/multichain/data/pdb_2021aug02/processed'\
                                            --single_chain False\
                                            --test False &

# clean single-chain data
nohup python utils/train_utils/data_utils   --clean_pdbs True \
                                            --data_path '/path/to/singlechain/data/'\
                                            --new_data_path '/path/to/singlechain/data/processed'\
                                            --single_chain True\
                                            --test False &
```

Now the data should be ready for use, and you can simply specifify the path in the config/train.yml file for each dataset

```yml
data:
  single_chain_data_path: /path/to/singlechain/data/processed
  multi_chain_data_path: /path/to/multichain/data/processed
```

##### Training
we have included an example HPC submission script which uses SLURM. you will have to make your own script if you are using a different job scheduler.

```bash
sbatch hpc_scripts/train.sh
```

But in general, you can start training like this if your machine has GPU(s) attached

```bash
nohup python learn_seqs.py > train.out 2> train.err &
```

note that this reads the configuration file, so make sure to edit it depending on your goal. it should be pretty self-explanatory.