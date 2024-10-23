# proteusAI
## Protein Sequence Prediction of Target Structure via Wave Function Encoding and Self-Attention

proteusAI is an AI that predicts the optimal protein sequence given a target structure of alpha-carbon ($C_a$) coordinates. 

Many protein sequence prediction AI models use contact maps, distance metrics, and/or dihedral angles of the protein structure as input features to AI models. However, these features fail to encode local AND global interactions of the $C_a$/backbone atoms in a concise and efficient way that the model can gain a reasonable inductive bias from. 

To achieve a greater inductive bias, we propose a method of encoding the three-dimensional coordinates of each token, i.e. each $C_a$, into high dimensional space through the use of wave functions. We model each $C_a$ as a point source via the Green's function to the Hemholtz equation in three dimension, and create a global wave function that is a superpositions of all $C_a$ atom point sources.

\psi_k = $\sum_{i=1}^N \frac{e^{ik|r-r_i|}}{4 \pi |r - r_i|}$

This method offers several advantages to existing methods. For one, 

, and some    
