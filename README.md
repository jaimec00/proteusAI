# proteusAI
## Protein Sequence Prediction of Target Structure via Wave Function Encoding and Self-Attention

proteusAI is a transformer model that predicts the optimal protein sequence given a target structure of alpha-carbon ($C_a$) coordinates. 

Many protein sequence prediction AI models use contact maps, distance metrics, and/or dihedral angles of the protein structure as input features to AI models. However, these features fail to encode local AND global interactions of the $C_a$/backbone atoms in a concise and efficient way that the model can gain a reasonable inductive bias from. 

To achieve a greater inductive bias, we propose a method of encoding the three-dimensional coordinates of each token, i.e. each $C_a$, into high dimensional feature space through the use of wave functions. We model each $C_a$ as a point source via the Green's function solution to the Hemholtz equation in three dimensions, and create a global wave function that is a superpositions of all $C_a$ atom point sources. More precisely, the global wavefunction, $\psi_k$ is defined as:

$\psi_k(r) = \sum_{i=1}^N \frac{e^{ik|r-r_i|}}{4 \pi |r - r_i|}$

where $|r - r_i|$ is Euclidaean norm of the positions vector of the $i_\text{th}$ $C_a$ source and the observer, i.e. the input to the wavefunction, and k is the wavenumber, related to the wavelength $\lambda$ by $k = \frac{2\pi}{\lambda}$.

Moreover, we can define multiple wavefunctions, each with a different k, and thus a different wavelength. In this case, wave functions corresponding to small $\lambda$ encode local interactions between the $C_a$ atoms, while larger $\lambda$ encodes global interactions. Thus, the output of a wave function, $\psi_k$, corresponds to two features of the input $C_a$, real part and imaginary part.

This method offers several advantages to existing methods. For one, it offers rotationally and translationally invariant representation of the protein, since the wave function only accounts for relative distances. Additionally, by using multiple wave functions of differing granularit (with different k), the model will capture a wide range of representations of the same structure, in which both local and global interactions are encoded. These features are passed to the transformer, which outputs amino acid probabilities for each position, and the model greedily selects the sequence. 
