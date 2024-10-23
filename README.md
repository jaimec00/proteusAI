#proteusAI

This is a transformer AI model that (potentially) uses a mixture of experts architecture to predict the optimal protein sequence of a given set of backbone or alpha carbon coordinate.

Proteins are downloaded from RCSB based on user specified filters. PDBs are cleaned according to the arguments passed. during training, pdbs are converted to Nx3 matrices and/or Nx3x3 matrices depending on whether the user specified to perform the training on alpha carbons, backbone atoms, or both. 
