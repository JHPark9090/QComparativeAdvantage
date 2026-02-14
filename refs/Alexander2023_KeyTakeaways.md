# Alexander et al. (2023) — Key Takeaways for Implementation

**Paper**: What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement
**Authors**: Yotam Alexander, Nimrod De La Vega, Noam Razin, Nadav Cohen
**Venue**: NeurIPS 2023 (Spotlight)
**arXiv**: [2303.11249](https://arxiv.org/abs/2303.11249)
**Code**: [github.com/nmd95/data_suitable_lc_nn_code](https://github.com/nmd95/data_suitable_lc_nn_code) (MIT License)


## 1. Core Idea

Locally connected neural networks (CNNs, LSTMs, local-attention Transformers) are mathematically equivalent to **Tree Tensor Networks**. As such, they satisfy an **Area Law** of entanglement — they cannot efficiently represent data distributions whose features have non-local (long-range) correlations. Data with high **Von Neumann entanglement entropy** follows a **Volume Law**, making it fundamentally unsuitable for these architectures.

Quantum circuits operate in the full Hilbert space and are **not** constrained by the Area Law, giving them a natural capacity advantage on high-entanglement data.


## 2. Theoretical Results

### Definition 1 — Quantum Entanglement (QE)

For a tensor $\mathcal{A} \in \mathbb{R}^{D_1 \times \cdots \times D_N}$ and a partition of axes $(\mathcal{K}, \mathcal{K}^c)$:

1. **Reshape** $\mathcal{A}$ as a matrix $[[\mathcal{A}; \mathcal{K}]]$ with rows indexed by $\mathcal{K}$ and columns by $\mathcal{K}^c$
2. **Compute SVD** singular values $\sigma_1 \geq \cdots \geq \sigma_D$
3. **Normalize** to a probability distribution: $\rho_d = \sigma_d^2 / \sum_{d'} \sigma_{d'}^2$
4. **Entropy**: $\text{QE}(\mathcal{A}; \mathcal{K}) = -\sum_d \rho_d \ln(\rho_d)$

### Definition 2 — Canonical Partitions

For $N = 2^L$ features, canonical partitions $\mathcal{C}_N$ are hierarchical, contiguous feature groupings matching the binary tree structure of the tensor network. At level $l$, features are split into $2^l$ contiguous blocks, and each block vs. its complement forms a partition.

### Empirical Data Tensor (Equation 4)

$$\mathcal{D}_{\text{emp}} := \frac{1}{M} \sum_{m=1}^{M} y^{(m)} \otimes \bigotimes_{n=1}^{N} \mathbf{x}^{(n,m)}$$

where $M$ = number of training samples, $y^{(m)} \in \{-1, +1\}$ = label, $\mathbf{x}^{(n,m)}$ = feature vector $n$ for sample $m$.

### Theorem 1 (Necessity)

If a locally connected tensor network with width $R$ approximates tensor $\mathcal{A}$ within error $\epsilon$, then for **all** canonical partitions:

$$\text{QE}(\mathcal{A}; \mathcal{K}) \leq \ln(R) + \frac{2\epsilon}{\|\mathcal{A}\|} \ln(D_{\mathcal{K}}) + 2\sqrt{\frac{2\epsilon}{\|\mathcal{A}\|}}$$

**Implication**: Locally connected networks can only represent low-entanglement tensors. Random tensors have entanglement $\sim \min(|\mathcal{K}|, |\mathcal{K}^c|)$, which is linear in $N$ — far exceeding $\ln(R)$.

### Theorem 2 (Sufficiency)

If for all canonical partitions:

$$\text{QE}(\mathcal{A}; \mathcal{K}) \leq \frac{\epsilon^2}{(2N-3) \|\mathcal{A}\|^2} \cdot \ln(R)$$

then the locally connected tensor network can fit $\mathcal{A}$ within error $\epsilon$.

### Corollary 1 (Prediction for Classification)

For binary SVM classification to be accurate, the empirical data tensor $\mathcal{D}_{\text{emp}}$ must have **low QE under all canonical partitions**. This is both necessary and sufficient.


## 3. Efficient Computation of QE from Data (Algorithm 2 / Appendix C)

### Problem

The empirical data tensor $\mathcal{D}_{\text{emp}}$ has size **exponential in $N$** (the number of features). Naive SVD is infeasible.

### Solution: Gram Matrix Approach

Based on Martyn et al. (2020), the algorithm avoids constructing $\mathcal{D}_{\text{emp}}$ entirely. Instead, it works with an $M \times M$ Gram matrix (where $M$ = number of samples).

**Complexity**: $\mathcal{O}(D \cdot N \cdot M^2 + M^3)$ — polynomial in all quantities.

### Step-by-Step Algorithm

Given $M$ labeled samples $\{(\mathbf{x}^{(m)}, y^{(m)})\}_{m=1}^M$ and a partition $(\mathcal{K}, \mathcal{K}^c)$:

**Step 1 — Angle Embedding**:
Each scalar feature $x_n$ is implicitly mapped to a 2D unit via a cosine kernel parameterized by angle $\theta$:

$$K(x_n^{(i)}, x_n^{(j)}) = \cos\bigl(\pi \theta \cdot (x_n^{(i)} - x_n^{(j)})\bigr)$$

**Step 2 — Gram Matrix for Partition $\mathcal{K}$**:
The Gram matrix is a product kernel over features in the partition:

$$G_{\mathcal{K}}[i, j] = \prod_{n \in \mathcal{K}} \cos\bigl(\pi \theta \cdot (x_n^{(i)} - x_n^{(j)})\bigr)$$

This is an $M \times M$ matrix. Computed analogously for $G_{\mathcal{K}^c}$.

**Step 3 — Incorporate Labels**:
Multiply the Gram matrix by the label outer product:

$$G_{\mathcal{K}} \leftarrow G_{\mathcal{K}} \odot (y^{(i)} \cdot y^{(j)})_{i,j}$$

where labels are mapped $\{0, 1\} \to \{-1, +1\}$.

**Step 4 — Vidal Decomposition**:
For each Gram matrix, compute the SVD-based square root:

$$G = U \Sigma U^T \implies V = \Sigma^{1/2} U^T$$

(With small noise added for numerical stability.)

**Step 5 — Cross-Partition SVD**:
Form the cross matrix and compute its SVD:

$$F = V_{\mathcal{K}} \cdot V_{\mathcal{K}^c}^T, \quad F = U_F \cdot D \cdot V_F^T$$

**Step 6 — Entanglement Entropy**:
From the singular values $d_1, \ldots, d_r$ of $F$:

$$c_k = \frac{d_k^2}{\sum_{k'} d_{k'}^2}, \qquad \text{QE} = -\sum_k c_k \ln(c_k)$$

### Batching

In practice, the computation is batched:
- **Data batches**: Randomly sample `batch_size` samples from $M$ total, repeat `n_batches` times, and average QE
- **Feature batches**: Within a partition, compute the product kernel in chunks of `fb_size` features to manage GPU memory


## 4. Canonical Partition Levels

For $N$ features indexed $0, \ldots, N-1$, the canonical partitions at level $l$ split features into $2^l$ contiguous chunks. For each chunk, the partition is (chunk, complement).

| Level | Number of Chunks | Partition Example ($N = 8$) |
|-------|------------------|-----------------------------|
| 1     | 2                | $\{0,1,2,3\}$ vs $\{4,5,6,7\}$ |
| 2     | 4                | $\{0,1\}$ vs $\{2,...,7\}$, $\{2,3\}$ vs $\{0,1,4,...,7\}$, etc. |
| 3     | 8                | $\{0\}$ vs $\{1,...,7\}$, $\{1\}$ vs $\{0,2,...,7\}$, etc. |

QE is **averaged** over all partitions at each level to produce a single entanglement score per level.


## 5. Area Law vs Volume Law

| Regime | Entanglement Scaling | Data Structure | Network Suitability |
|--------|---------------------|----------------|---------------------|
| **Area Law** | $\text{QE} \sim \mathcal{O}(\ln R)$ — sublinear in partition size | Local feature correlations (natural images, speech) | Suitable for LCNNs |
| **Volume Law** | $\text{QE} \sim \mathcal{O}(\min(|\mathcal{K}|, |\mathcal{K}^c|))$ — linear in partition size | Non-local correlations (shuffled data, random tensors) | **Not suitable** for LCNNs; requires global architectures |

**Key insight**: Feature shuffling destroys locality, pushing data from Area Law toward Volume Law regime.


## 6. Surrogate Entanglement (Definition 4)

For computational efficiency in feature reordering, the paper introduces a surrogate metric based on the **multivariate Pearson correlation** (Puccetti 2022):

$$\text{Surrogate}(\mathcal{K}, \mathcal{K}^c) = \sum_{n \in \mathcal{K}, n' \in \mathcal{K}^c} |\text{Corr}(x_n, x_{n'})|$$

This sums the absolute pairwise correlations between features across the partition boundary. It is strongly correlated with true entanglement entropy (Appendix E) and enables framing the feature reordering problem as a **minimum balanced graph cut** problem, solvable with METIS or similar tools.


## 7. Feature Reordering (Data Preprocessing)

The paper proposes rearranging feature indices to **minimize** entanglement under canonical partitions:

1. Compute the correlation matrix $C_{nn'} = |\text{Corr}(x_n, x_{n'})|$ over all feature pairs
2. Treat this as a weighted graph
3. Recursively solve **minimum balanced cut** problems (using METIS) to find a feature ordering that places highly correlated features adjacently
4. This reduces entanglement and improves LCNN performance on shuffled/tabular data


## 8. Experimental Validation

### Datasets Tested
- **Audio**: Speech Commands (randomly permuted time steps)
- **Tabular**: Standard tabular benchmarks
- **Images**: CIFAR-10 (randomly permuted pixels)

### Key Findings
1. **Inverse relationship**: Higher entanglement entropy ↔ lower LCNN accuracy (across CNNs, RNNs, local self-attention)
2. **Feature reordering works**: Rearranging features via the surrogate metric substantially improves test accuracy on permuted datasets
3. **Consistent across architectures**: The entanglement-accuracy relationship holds for all locally connected architectures tested


## 9. Relevance to QComparativeAdvantage Project

### What We Adopt
1. **Feature Shuffling Protocol**: Randomly permute $k\%$ of feature indices ($k = 0, 10, \ldots, 100$) to exogenously increase QE
2. **QE Measurement**: Use their `EntropyCalc` class (or reimplement the Gram matrix algorithm) to verify monotonic QE increase with shuffling percentage
3. **Theoretical justification**: Their Theorems 1-2 provide the mathematical basis for why classical LCNNs degrade on high-entanglement data

### What We Extend
1. **Quantum counterpart comparison**: They only study classical LCNNs; we add quantum models (QCNN, QLSTM, QTransformer) operating in full Hilbert space
2. **Comparative Advantage regression**: We quantify the *relative* degradation rates via interaction regression ($\beta_2$), not just absolute performance
3. **Economic framework**: We frame the crossover as Comparative Advantage (Acemoglu & Restrepo 2018), providing a policy-relevant lens


## 10. Implementation Notes

### From Their Codebase (`nmd95/data_suitable_lc_nn_code`)

| Component | File | What It Does |
|-----------|------|-------------|
| `EntropyCalc` | `entropy/entropy_module.py` | Gram matrix computation, Vidal decomposition, QE calculation |
| `measure_entropy.py` | `entropy/measure_entropy.py` | CLI entry point for entropy measurement from YAML config |
| Feature reordering | `rearrangement_algorithms/` | METIS-based minimum balanced cut |
| Configs | `configs/` | YAML files for experiment parameters |

### Key Parameters
- `theta`: Angle embedding parameter (fraction of $\pi$). Controls kernel bandwidth.
- `batch_size`: Number of samples per Gram matrix computation (typically 100-500).
- `n_batches`: Number of random batches to average over.
- `fb_size`: Feature batch size (for memory-efficient Gram matrix computation).
- `vidal_noise`: Small noise for SVD numerical stability.
- `levels`: Which canonical partition levels to compute QE at.

### Dependencies
- PyTorch (GPU-accelerated Gram matrix computation)
- NumPy
- METIS (for feature reordering; optional for our use case)
