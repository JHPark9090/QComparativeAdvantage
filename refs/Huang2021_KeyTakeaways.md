# Huang et al. (2021) — Key Takeaways for Implementation

**Paper**: Power of Data in Quantum Machine Learning
**Authors**: Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, Jarrod R. McClean
**Venue**: Nature Communications, 12, 2631
**arXiv**: [2011.01938](https://arxiv.org/abs/2011.01938)
**Code**: TensorFlow Quantum (integrated into experiments)


## 1. Core Idea

**Data provides computational power.** Problems that are classically hard to compute from scratch (BQP-hard) can become classically easy to *predict from data*. This means quantum advantage in ML is more nuanced than computational complexity alone suggests — the relevant question is not "can a classical computer simulate this quantum circuit?" but rather "given training data, can classical ML match quantum ML in prediction accuracy?"

The paper provides a rigorous framework based on **geometric difference** ($g$) to determine *when* quantum advantage in prediction is possible and *when* it is not.


## 2. Theoretical Results

### Setup: Quantum Target Function

$$f(x) = \text{Tr}(O^U \cdot \rho(x))$$

where $\rho(x)$ is the encoded quantum state, $O^U = U_{\text{QNN}}^\dagger O U_{\text{QNN}}$ is the rotated observable, and $f(x)$ is a quadratic function in the density matrix entries.

### Proposition 1 — Hardness Without Data

If a classical algorithm *without training data* can compute $f(x)$ efficiently for any $U_{\text{QNN}}$ and $O$, then BPP = BQP.

**But**: With $N \propto p^2/\epsilon^2$ training samples (where $p = 2^n$), a classical model achieves prediction error $\epsilon$. Data "elevates" classical models.

### Prediction Error Bound (Equation 3)

For a kernel-based model $h(x) = w^\dagger \phi(x)$ with kernel $K$:

$$\mathbb{E}_{x \sim \mathcal{D}} |h(x) - f(x)| \leq c \sqrt{\frac{s_K(N)}{N}}$$

where $s_K(N) = \|w\|^2$ is the **model complexity** after training on $N$ samples.

### Definition — Geometric Difference (Equation 5)

$$g(K^1 \| K^2) = \sqrt{\left\| \sqrt{K^2} \cdot (K^1)^{-1} \cdot \sqrt{K^2} \right\|_\infty}$$

where:
- $K^1, K^2$ are $N \times N$ kernel (Gram) matrices
- $\|\cdot\|_\infty$ is the spectral norm (largest singular value)
- $\text{Tr}(K^1) = \text{Tr}(K^2) = N$ (normalized traces)

**Properties**:
- Asymmetric: $g(K^1 \| K^2) \neq g(K^2 \| K^1)$ in general
- Computable classically in $\mathcal{O}(N^3)$ via SVD
- Label-independent — only depends on the input data and kernels

### The Fundamental Inequality (Equation 6)

$$s_{K^1} \leq g(K^1 \| K^2)^2 \cdot s_{K^2}$$

**Applied to classical vs. quantum**: Let $g_{CQ} = g(K^C \| K^Q)$. Then:

$$\text{Classical prediction error} \leq g_{CQ} \times \text{Quantum prediction error}$$

- If $g_{CQ} \approx 1$: **No quantum advantage** for any function/dataset
- If $g_{CQ} \sim \sqrt{N}$: **Potential quantum advantage** exists

### Effective Dimension (Equation 7)

$$d = \dim(P_Q) = \text{rank}(K^Q) \leq N$$

where $P_Q$ is the projector onto the subspace spanned by $\{\text{vec}(\rho(x_1)), \ldots, \text{vec}(\rho(x_N))\}$.

### Prediction Bound for Quantum Kernel (Equation 8)

$$\mathbb{E}|h(x) - f(x)| \leq c \sqrt{\frac{\min(d, \text{Tr}(O^2))}{N}}$$


## 3. Three-Step Flowchart for Assessing Quantum Advantage

### Step 1 — Geometry Test (label-independent)

Compute $g_{CQ} = g(K^C \| K^Q)$.
- $g_{CQ} \ll \sqrt{N}$ → **Classical wins** — no quantum advantage possible for any function
- $g_{CQ} \sim \sqrt{N}$ → Proceed to Step 2

### Step 2 — Dimensionality Test

Compute $d = \text{rank}(K^Q)$.
- $d \ll N$ → **Classical wins** — low effective dimension means a classical surrogate can match quantum
- Otherwise → Proceed to Step 3

### Step 3 — Complexity Test (label-dependent)

Compute model complexities $s_C$ and $s_Q$ using actual labels.
- $s_C \sim N$ and $s_Q \ll N$ → **Quantum advantage**
- $s_C \ll N$ → **Classical wins** regardless
- $s_Q \sim N$ → **Both struggle**


## 4. Projected Quantum Kernel (PQK)

The standard quantum kernel $k^Q(x_i, x_j) = \text{Tr}(\rho(x_i) \rho(x_j))$ lives in exponentially large Hilbert space, but paradoxically the kernel matrix can approach a scaled identity at large qubit counts, losing discriminative power and making $g$ small.

**Solution** — Project to local reduced density matrices (Equation 9):

$$k^{PQ}(x_i, x_j) = \exp\left(-\gamma \sum_k \|\rho_k(x_i) - \rho_k(x_j)\|_F^2\right)$$

where:
- $\rho_k(x) = \text{Tr}_{j \neq k}[\rho(x)]$ is the **1-particle reduced density matrix** (1-RDM) of qubit $k$
- $\gamma$ is a bandwidth hyperparameter
- Sum runs over all qubits

**Why it works**: Inspired by density functional theory (Hohenberg-Kohn). Even one-body densities can capture essential many-body correlations. The PQK achieves **much larger geometric difference** from classical kernels.


## 5. Key Experimental Findings

### Three Quantum Embeddings Tested

| Embedding | Type | Entanglement | $g$ from Classical |
|-----------|------|-------------|-------------------|
| E1 | Separable rotations | None (product state) | Small |
| E2 | IQP-type (ZZ interactions) | Moderate | Moderate |
| E3 | Hamiltonian evolution | High | **Large** |

### Results on Fashion-MNIST + Quantum Labels

| Scenario | Result |
|----------|--------|
| E1 + standard quantum kernel | Classical ML competitive |
| E2 + standard quantum kernel | Classical ML competitive |
| E3 + **projected** quantum kernel | **>20% quantum prediction advantage** |
| Engineered datasets (maximize $g$) | Robust quantum advantage across all classical baselines |

### Classical Baselines (All Hyperparameter-Tuned)
- Random Forest, Gradient Boosting, AdaBoost
- Gaussian (RBF) kernel SVM, Linear model
- CNN, Feedforward NN
- **All outperformed** by PQK on E3 embedding with engineered datasets

### Scale
- Simulations up to **30 qubits** (among the largest gate-based QML experiments at publication)
- Peak throughput ~1.1 petaflop/s using TensorFlow Quantum


## 6. Kernel Methods vs. Variational Circuits

The paper establishes that supervised quantum models are equivalent to kernel methods (following Schuld 2021):
- The encoding circuit $\rho(x)$ defines the kernel
- The variational ansatz $U_{\text{QNN}}$ determines which function in the RKHS is learned, but does **not** change the kernel
- Therefore, model complexity $s_K$ and geometric difference $g$ capture the **fundamental capability of the data encoding**, independent of the variational ansatz
- Kernel training is convex — avoids barren plateau issues


## 7. Relevance to QComparativeAdvantage Project

### What We Adopt

1. **Geometric Difference ($g$) as Mechanism Metric**: Computable on $N \times N$ matrices, label-independent. Measures whether quantum and classical models define fundamentally different data geometries.
2. **Three-Step Flowchart**: Provides a principled way to diagnose *why* quantum advantage appears or doesn't at each entanglement level.
3. **Model Complexity ($s_K$)**: Can be computed for both classical and quantum kernels to understand generalization behavior.

### Our Hypothesis

As data entanglement (QE from Alexander et al.) increases via feature shuffling:
1. Classical kernel geometry becomes **less aligned** with the data → $s_C$ increases
2. Quantum kernel geometry remains aligned → $s_Q$ stays low
3. Therefore $g_{CQ}$ should **increase monotonically** with QE
4. This produces the interaction effect $\beta_2 > 0$ in our regression model

### What We Extend

1. **Entanglement-dependent $g$**: Huang et al. use fixed datasets. We compute $g_{CQ}$ across the entanglement spectrum ($D_0, \ldots, D_{100}$) and show it tracks QE.
2. **Architecture pairs, not just kernels**: We test CNN/QCNN, LSTM/QLSTM, Transformer/QTransformer — going beyond the kernel framework to variational circuits.
3. **Trade-theoretic interpretation**: $g_{CQ}$ maps to "task complexity" in the Acemoglu-Restrepo framework — higher $g$ means the task has shifted to quantum's comparative advantage region.


## 8. Implementation Notes

### Computing $g(K^C \| K^Q)$

```python
import numpy as np

def geometric_difference(K_C, K_Q):
    """
    Compute g(K^C || K^Q) = sqrt(||sqrt(K^Q) @ inv(K^C) @ sqrt(K^Q)||_inf)

    Args:
        K_C: (N, N) classical kernel matrix, Tr(K_C) = N
        K_Q: (N, N) quantum kernel matrix, Tr(K_Q) = N
    Returns:
        g: geometric difference (scalar)
    """
    # Normalize traces
    N = K_C.shape[0]
    K_C = K_C * N / np.trace(K_C)
    K_Q = K_Q * N / np.trace(K_Q)

    # Regularize for numerical stability
    K_C_reg = K_C + 1e-8 * np.eye(N)

    # Compute sqrt(K_Q)
    eigvals_Q, eigvecs_Q = np.linalg.eigh(K_Q)
    eigvals_Q = np.maximum(eigvals_Q, 0)
    sqrt_K_Q = eigvecs_Q @ np.diag(np.sqrt(eigvals_Q)) @ eigvecs_Q.T

    # Compute inv(K_C)
    K_C_inv = np.linalg.inv(K_C_reg)

    # Compute the matrix product
    M = sqrt_K_Q @ K_C_inv @ sqrt_K_Q

    # Spectral norm (largest singular value)
    spectral_norm = np.linalg.norm(M, ord=2)

    return np.sqrt(spectral_norm)
```

### Computing Model Complexity ($s_K$)

```python
def model_complexity(K, y):
    """
    Compute s_K = w^T w where w = K^{-1} y (kernel regression solution).

    Args:
        K: (N, N) kernel matrix
        y: (N,) labels
    Returns:
        s_K: model complexity (scalar)
    """
    K_reg = K + 1e-8 * np.eye(K.shape[0])
    w = np.linalg.solve(K_reg, y)
    return np.dot(w, K @ w)
```

### Projected Quantum Kernel (PQK)

```python
def projected_quantum_kernel(rdms_i, rdms_j, gamma=1.0):
    """
    Compute k^PQ(x_i, x_j) = exp(-gamma * sum_k ||rho_k(x_i) - rho_k(x_j)||_F^2)

    Args:
        rdms_i: list of 1-RDMs for input x_i (one per qubit)
        rdms_j: list of 1-RDMs for input x_j (one per qubit)
        gamma: bandwidth parameter
    Returns:
        kernel value (scalar)
    """
    dist_sq = sum(
        np.linalg.norm(rho_i - rho_j, 'fro') ** 2
        for rho_i, rho_j in zip(rdms_i, rdms_j)
    )
    return np.exp(-gamma * dist_sq)
```

### Key Parameters

- `N`: Number of training samples (typically 100–1000 for kernel experiments)
- `gamma`: PQK bandwidth (tune via cross-validation)
- Regularization: Add $\epsilon I$ to kernel matrices before inversion ($\epsilon \sim 10^{-8}$)
- Trace normalization: $\text{Tr}(K) = N$ required for valid $g$ computation


## 9. Connection to Other Papers in Our Framework

| Paper | Metric | What It Measures | Role |
|-------|--------|-----------------|------|
| Alexander et al. (2023) | Von Neumann Entropy (QE) | Data entanglement — non-local feature correlations | **Independent variable** |
| Abbas et al. (2021) | Effective Dimension ($d_{\text{eff}}$) | Model capacity per parameter (FIM-based) | **Mechanism: capacity** |
| **Huang et al. (2021)** | Geometric Difference ($g$) | Kernel geometry mismatch between classical and quantum | **Mechanism: geometry** |
| Martin & Mahoney (2021) | HTSR $\alpha$ | Generalization quality via heavy-tailed spectral analysis | **Mechanism: generalization** |

### Expected Behavior Across Entanglement Spectrum

| QE Level | $d_{\text{eff}}$ (Classical) | $d_{\text{eff}}$ (Quantum) | $g_{CQ}$ | Classical Accuracy | Quantum Accuracy |
|----------|------------------------------|----------------------------|-----------|-------------------|-----------------|
| Low (0%) | High | Similar | Small (~1) | High | Similar (slightly lower) |
| Medium (50%) | Declining | Maintained | Growing | Declining | Slower decline |
| High (100%) | Low | Maintained | Large ($\sim\sqrt{N}$) | Poor | Better (crossover) |
