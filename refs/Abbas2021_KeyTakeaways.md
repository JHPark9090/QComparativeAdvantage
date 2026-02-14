# Abbas et al. (2021) — Key Takeaways for Implementation

**Paper**: The Power of Quantum Neural Networks
**Authors**: Amira Abbas, David Sutter, Christa Zoufal, Aurelien Lucchi, Alessio Figalli, Stefan Woerner
**Venue**: Nature Computational Science, 1(6), 403–409
**arXiv**: [2011.00027](https://arxiv.org/abs/2011.00027)
**Follow-up**: [2112.04807](https://arxiv.org/abs/2112.04807) (effective dimension theory extension)


## 1. Core Idea

Quantum neural networks (QNNs) can be compared with classical neural networks on a **fair, information-geometric footing** via the **Fisher Information Matrix (FIM)**. The FIM captures how sensitively a model's output distribution changes with parameter perturbations. From the FIM, the **Effective Dimension** ($d_{\text{eff}}$) quantifies the true capacity of a model — how many of its $d$ parameters are "actively used." QNNs with hard (classically non-simulable) feature maps achieve higher effective dimension than parameter-matched classical networks, indicating greater model capacity.


## 2. Theoretical Results

### Fisher Information Matrix (FIM)

For a statistical model $p(x, y; \theta)$ with parameters $\theta \in \Theta \subseteq [-1, 1]^d$:

$$F(\theta) = \mathbb{E}_{(x,y) \sim p} \left[ \nabla_\theta \log p(x, y; \theta) \cdot \nabla_\theta \log p(x, y; \theta)^T \right] \in \mathbb{R}^{d \times d}$$

**Empirical approximation** with $k$ i.i.d. samples:

$$\tilde{F}_k(\theta) = \frac{1}{k} \sum_{j=1}^{k} \nabla_\theta \log p(x_j, y_j; \theta) \cdot \nabla_\theta \log p(x_j, y_j; \theta)^T$$

### Normalized FIM ($\hat{F}$)

To enable **scale-invariant** comparison across model families:

$$\hat{F}_{ij}(\theta) = \frac{d}{V_\Theta \cdot \int_\Theta \text{tr}(F(\theta)) \, d\theta} \cdot F_{ij}(\theta)$$

where $V_\Theta = \int_\Theta d\theta$ is the volume of the parameter space. This ensures $\text{tr}(\hat{F})$ averages to $d$ over $\Theta$.

### Definition 1 — Effective Dimension

$$d_{\gamma, n}(\mathcal{M}_\Theta) = \frac{2 \log \left( \frac{1}{V_\Theta} \int_\Theta \sqrt{\det\left( I_d + \frac{\gamma n}{2\pi \log n} \hat{F}(\theta) \right)} \, d\theta \right)}{\log \left( \frac{\gamma n}{2\pi \log n} \right)}$$

where:
- $I_d$ = identity matrix
- $\gamma \in (0, 1]$ = scaling parameter
- $n$ = number of data samples

**Key properties (Remark 2)**:
- As $n \to \infty$: $d_{\text{eff}} \to \bar{r} = \max_{\theta \in \Theta} \text{rank}(F(\theta))$
- $d_{\text{eff}} \in [0, d]$ — bounded by the parameter count
- Convergence speed depends on how evenly spread the FIM eigenvalues are

### Theorem 3 — Generalization Bound

If $\hat{F}(\theta)$ has full rank for all $\theta$ and $\|\nabla_\theta \log \hat{F}\| \leq \Lambda$, then for loss bounded in $[-B/2, B/2]$:

$$P\left( \sup_\theta |R(\theta) - R_n(\theta)| \geq \epsilon \right) \leq c_{d,\Lambda} \cdot \left(\frac{\gamma n^{1/\alpha}}{2\pi \log(n)^{1/\alpha}}\right)^{d_{\gamma,n^{1/\alpha}}/2} \cdot \exp\left(-\frac{16M \cdot 2\pi \log n}{B^2 \gamma}\right)$$

**Implication**: The effective dimension plays a role analogous to VC dimension — it controls the generalization gap between training and test error.

### Connection to Barren Plateaus

- FIM eigenvalue spectrum concentrated near zero → barren plateau (exponentially flat optimization landscape)
- Evenly spread FIM spectrum → good trainability
- The effective dimension provides a unified lens for both **capacity** and **trainability**


## 3. The Three-Model Comparison

| Model | Feature Map | FIM Properties | Effective Dimension |
|-------|------------|----------------|---------------------|
| Classical Feedforward NN | N/A (standard layers) | Highly degenerate eigenvalues | Lowest |
| "Easy" QNN | Classically simulable | Concentrates near zero (barren plateau) | Medium |
| "Hard" QNN | Havlicek et al. (2019) — not classically simulable | Non-degenerate, evenly spread | **Highest** |

### Hard Feature Map (Havlicek et al. 2019)

1. Hadamard gates on all qubits
2. $R_Z(x_i)$ gates per qubit (data encoding)
3. $R_{ZZ}(x_i \cdot x_j)$ gates for all qubit pairs (interaction encoding)
4. Layers 2–3 repeated once


## 4. Key Findings

1. **Higher capacity**: The hard QNN achieves significantly higher normalized effective dimension ($d_{\text{eff}}/d$) than parameter-matched classical models across all finite data sizes
2. **Faster training**: QNN reaches lower loss in fewer iterations (averaged over 100 trials), attributed to better FIM spectrum
3. **Data encoding is critical**: The feature map choice determines whether barren plateaus arise. Hard (non-simulable) feature maps show resilience
4. **Hardware validation**: IBM `ibmq_montreal` confirms the advantage is not a simulation artifact
5. **Convergence hierarchy**: QNN needs **less data** to reach full capacity ($d_{\text{eff}} \to \bar{r}$)


## 5. Relationship to QComparativeAdvantage Project

### What We Adopt

1. **Effective Dimension as Mechanism Metric**: $d_{\text{eff}}$ quantifies *why* quantum models maintain accuracy on high-entanglement data — they have higher model capacity per parameter
2. **FIM-based analysis**: The Fisher Information Matrix provides a model-agnostic comparison framework applicable to both classical and quantum architectures
3. **Generalization bound interpretation**: Their Theorem 3 provides theoretical grounding for why higher capacity doesn't automatically mean overfitting — it depends on the data-model match

### How We Extend

1. **Interaction with data entanglement**: Abbas et al. measure $d_{\text{eff}}$ on fixed datasets. We measure how $d_{\text{eff}}$ changes across the entanglement spectrum ($D_0, D_{10}, \ldots, D_{100}$)
2. **Multiple architecture pairs**: They compare one classical + two quantum models. We compare three matched pairs (CNN/QCNN, LSTM/QLSTM, Transformer/QTransformer)
3. **Comparative Advantage framing**: We use $d_{\text{eff}}$ as a *mechanism variable* in the interaction regression $\text{Accuracy}_{tm} = \beta_0 + \beta_1 \cdot Q_m + \beta_2 \cdot (Q_m \times \text{QE}_t) + \varepsilon$


## 6. Computation Methods

### Method A: Full FIM via Parameter-Shift Rule (Small Circuits)

For quantum circuits with $\leq 100$ parameters:

**Quantum Fisher Information**:
$$F_{ij} = 4 \cdot \text{Re}\left( \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle \right)$$

**Parameter-shift gradient**:
$$|\partial_i \psi\rangle \approx \frac{|\psi(\theta + \frac{\pi}{2} e_i)\rangle - |\psi(\theta - \frac{\pi}{2} e_i)\rangle}{2}$$

**Complexity**: $\mathcal{O}(d)$ circuit evaluations per parameter, $\mathcal{O}(d^2)$ for full FIM

### Method B: Sampling-Based Estimation (Large Circuits)

For circuits with $> 100$ parameters:
1. Sample random parameter directions
2. Estimate diagonal FIM entries
3. Use trace and Frobenius norm approximations

### Simplified Effective Dimension (Our Existing Implementation)

Our `measure_comprehensive_metrics.py` uses a simplified version:

$$d_{\text{eff}} = \frac{\text{Tr}(F)}{\|F\|_F}$$

This is a **ratio of the trace to the Frobenius norm** of the FIM, which captures the "effective rank" — how many eigenvalues contribute meaningfully to the trace. It ranges from 1 (single dominant eigenvalue) to $d$ (all eigenvalues equal).

**Note**: This differs from the full Abbas et al. formula (Definition 1) which incorporates data sample size $n$ and the determinant. For the full comparative advantage analysis, we should implement the complete formula.


## 7. Implementation Notes

### From Our Existing Codebase (`QuantumDilatedCNN_Analysis/scripts/measure_comprehensive_metrics.py`)

| Component | Function | What It Does |
|-----------|----------|-------------|
| `effective_dimension_fisher()` | Line 262 | Full FIM computation + $d_{\text{eff}} = \text{Tr}(F)/\|F\|_F$ |
| `compute_fisher_matrix_qml()` | Line 343 | QFI via parameter-shift rule |
| `effective_dimension_sampling()` | (fallback) | Sampling-based for $d > 100$ params |
| `expressibility()` | Line 186 | KL/JS divergence from Haar distribution |
| `meyer_wallach_measure()` | Line 62 | $Q = 2(1 - \frac{1}{n}\sum \text{purities})$ |

### What Needs to Be Added for QComparativeAdvantage

1. **Full Abbas et al. effective dimension** (Definition 1) — with log-determinant formula depending on $n$
2. **Normalized FIM** $\hat{F}$ — average trace over parameter space for normalization
3. **Data-dependent FIM** — compute FIM using actual data distributions $p(y|x;\theta)$, not just quantum state overlap
4. **Eigenvalue spectrum analysis** — histogram of FIM eigenvalues to diagnose barren plateaus

### Key Parameters

- `n_samples`: Number of parameter samples to average FIM over (typically 50–200)
- `n`: Data sample count (varies for scaling analysis)
- `gamma`: Scaling parameter (default 1.0)
- `shift`: Parameter-shift value ($\pi/2$ for standard Pauli rotation gates)


## 8. Experimental Validation (Their Paper)

| Dataset | Models | Key Result |
|---------|--------|------------|
| Iris (4 features, 2 classes) | Classical FF, Easy QNN, Hard QNN | Hard QNN: highest $d_{\text{eff}}$, fastest training |
| Fashion-MNIST (subset) | Same | Same ordering confirmed |
| IBM `ibmq_montreal` (hardware) | Hard QNN only | ~33 iterations to convergence, confirms advantage on real hardware |

**All models had the same number of trainable parameters** for fair comparison.
