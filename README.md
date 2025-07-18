# proteusAI
## Protein Sequence Prediction of Target Structure via Wave Function Embedding and Multi-Scale Geometric Flash-Attention

proteusAI is a transformer model that predicts the optimal protein sequence given a target structure of alpha-carbon ($C_a$) coordinates. 

Many protein sequence prediction AI models use contact maps, distance metrics, and/or dihedral angles of the protein structure as input features to AI models. However, these features fail to encode local AND global interactions of the $C_a$/backbone atoms in a concise and efficient way that the model can gain a reasonable inductive bias from. 

To achieve a greater inductive bias, we propose a method of encoding the three-dimensional coordinates of each token, i.e. each $C_a$, into high dimensional feature space through the use of wave functions. We model each $C_a$ as a point source via the Green's function solution to the Hemholtz equation in three dimensions, and create a global wave function that is a superpositions of all $C_a$ atom point sources. More precisely, the global wavefunction, $\psi_k$ is defined as:

$\psi_k(r) = \sum_{j=0}^N \frac{e^{ik|r-r_j|}}{|r - r_j|}$

where $|r - r_j|$ is Euclidaean norm of the positions vector of the $j^\text{th}$ $C_a$ source and the observer, i.e. the input to the wavefunction, and k is the wavenumber, related to the wavelength $\lambda$ by $k = \frac{2\pi}{\lambda}$.

Moreover, we can define multiple wavefunctions, each with a different k, and thus a different wavelength. In this case, wave functions corresponding to small $\lambda$ encode local interactions between the $C_a$ atoms, while larger $\lambda$ encode global interactions. Thus, the output of a wave function, $\psi_k$, corresponds to two features of the input $C_a$, a real part and imaginary part, i.e. a cos and sin term. To emphasize local interactions, since these are more prone to large fluctuations from small changes in wavelength, the wavelengths are sampled logarithmically from $\lambda_{min}$ to $\lambda_{max}$, given a base, $b$. This gives the general wave function featurization formula, termed Wavefunction Embedding (WE):

$WE(2i, r) = \sum_{j=0}^N \frac{1}{{|r-r_j|}} cos( k_{i} |r-r_j| ) $

$WE(2i+1, r) = \sum_{j=0}^N \frac{1}{|r-r_j|} sin( k_{i} |r-r_j| ) $

Where, 

$k_{i} = \frac{2\pi}{\lambda_{i}}$

$\lambda_{i} = \lambda_{min} + (\lambda_{max}-\lambda_{min})(\frac{ b^{ 2i/d_{model} } - 1 } {b - 1} )$

Note the similarity between this formula and the traditional positional encoding formula:

$PE(2i, p) = sin(\frac{p}{10000^{2i/d_{model}}})$

$PE(2i+1, p) = cos(\frac{p}{10000^{2i/d_{model}}})$

This is because the wave function embedding process can be seen as a generalization of positional encoding for irregularly spaced tokens in arbitrary dimensions.

This method offers several advantages to existing methods. For one, it offers rotationally and translationally invariant representation of the protein, since the wave function only accounts for relative distances. Additionally, by using multiple wave functions of differing granularity (with different k), the model will capture a wide range of representations of the same structure, in which both local and global interactions are encoded. While computing the superposed wave function outputs for each Ca, and for each of the d_model//2 wave functions, scales O($N^2$) in compute, memory, and time, we have implemented a custom cuda kernel that fuses all of the operations into a single GPU kernel, which significantly speeds up the computation and drastically reduces memory usage. For batch size of 1, sequence length of 16384, and d_model of 512, the kernel runs in 386 ms on h100 GPU, while a naive pytorch version takes minutes, due to the necessary tiling for an input of that size.

Additionally, the Wavefunction Embedding module implements an extremely efficient backwards pass, achieving 10X speedup and 1000X memory reduction WITHOUT any hardware optimizations, written fully in PyTorch. This is achieved by storing the sums of the cosine terms for each token and the sum of the sin terms during the forward pass, each of which is only batch x N x d_model//2. this avoids both storing large intermediate tensors and recomputation, and is accomplished by analytically simplifying the gradient computation, dropping the computational complexity of the backward pass from $O(N^2)$ in to $O(N)$. This allows the function to compute the gradients with respect to the wavenumbers, which makes it possible to make the wavelengths themselved to be learnable, allowing end to end differentiability, adaptive and interpretable featurization, and requiring only the coordinates to be served to the model as input.

The wavefunction features are combined with ESM2 amino acid embeddings which contain rich evolutionary and structural information, by adding the two features together. The resulting features align very well with the rest of the model, which is a stack of traditional Encoder layers from the original transformer paper. While transformers are known for their ability to perform long range attention, it is still beneficial to inject a spatial bias into the model, so that extremely distant residues do not affect each other too much. To solve this problem, we introduce Geometric Attention, which is a novel multi-head attention (MHA) mechanism. In the custom MHA module, the attention logits are scaled by Radial Basis Functions (RBF). Each head of the MHA module gets assigned a specific spread ($\sigma_{head}$) to compute the RBFs. The RBF is thus:

$RBF_{head}(r_q, r_k) = 1 + exp(-\frac{|r_q-r_k|^2}{2\sigma_{head}^2})$

Where $r_q$ is the physical position of the token corresponding to the query matrix row, and $r_k$ is the physical position of the token corresponding to the key matrix (transposed) column. The added 1 to the RBF serves to stabilize gradient for Q and K, but still ensures multiplicative interactions between the RBF and the attention logits, rather than adding a bias, which leads to cross talk between the RBFs and the attention logits in the backwards pass, allowing the Q and K matrices to learn directly from the geometry of the structures.

To reduce the memory footprint and speed up the computation, the Multi-Scale Geometric Attention module is fused into a single GPU kernel using triton, taking inspiration from the Flash Attention 2 paper (https://arxiv.org/abs/2205.14135). A custom backwards pass is also implemented to make not only Q, K, and V learnable, but also the spread of each attention head. Thus, each head learns at what scale it should evaluate the RBFs, and how to weigh pairs of tokens. This design aligns very well with the previously described featurization process, since the features themselves correspond to different representations of the same structure at distinct scales via the learned wavelength ranges ($\lambda_{min} and \lambda_{max}$) and distributions ($b$). since each head operates on a well defined feature space, $d_k = d_{model}/n_{heads}$, the heads can learn what scale to evaluate the features they are working on.

This multi-scale geometric attention mechanism can be seen as a generalization of graph neural networks (GNN), since the scaled attention mechanism creates soft, continuous edges between token pairs, which are defined at multiple scales. 

After passing through all Encoder layers, the logits pass through a linear layer to convert the $d_{model}$ feature space into AA feature space (20 dimensions, one for each amino acid) and softmax is performed to get amino acid probabilities for each position. The model selects the most confident prediction and auto-regressively updates the sequence, until the final prediction is reached. 
