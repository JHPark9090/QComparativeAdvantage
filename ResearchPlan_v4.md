# Research Plan v4: The Division of Labor in Intelligence — Quantifying the Comparative Advantage of Quantum Machine Learning via Data Entanglement

## 1. Abstract / Executive Summary
* **Objective**: To empirically demonstrate that while Classical Machine Learning (CML) holds an **Absolute Advantage** in general tasks, Quantum Machine Learning (QML) holds a **Comparative Advantage** in tasks exhibiting high **Quantum Entanglement ($QE$)**.
* **Theoretical Basis**: We synthesize the **Task-Based Model of Automation** (Acemoglu & Restrepo, 2018) with the **Entanglement Limit of Locally Connected Neural Networks** (Alexander et al., 2023).
* **Methodology**: We utilize a "Feature Shuffling Protocol" to exogenously vary the Entanglement Entropy of standard datasets across two domains — spatial (MNIST, CIFAR-10) and temporal (PhysioNet EEG, ETTh1, Weather). We compare the performance decay rates of "Local" Classical Agents (CNN, TCN, Local-Attention Transformer) against "Global" Quantum Agents (QCNN, QTCN, QTSTransformer).
* **Expected Outcome**: We predict a "Crossover" in efficiency where QML degradation is flatter than CML degradation as Entanglement rises, justifying a hybrid "Gains from Trade" computing architecture.


## 2. Theoretical Framework (The "Why")

### 2.1 The Economic Model: Task-Based Production
**Source**: Acemoglu, D., & Restrepo, P. (2018). The Race Between Man and Machine. _American Economic Review._

* **Concept**: Intelligence production is defined as a continuum of tasks $i \in [0, 1]$ indexed by complexity.
* **Application**: We posit that Classical Productivity $A_C(i)$ decays exponentially with task complexity ($F_t$), while Quantum Productivity $A_Q(i)$ decays polynomially or remains stable.
* **Condition**: Comparative Advantage exists if $\frac{A_C(i)}{A_Q(i)} > \frac{A_C(j)}{A_Q(j)}$ for low-complexity task $i$ and high-complexity task $j$.


### 2.2 The Physical Mechanism: Data Entanglement
**Source**: Alexander, Y., et al. (2023). What Makes Data Suitable for a Locally Connected Neural Network? _NeurIPS._

* **Theorem**: Locally Connected Neural Networks (LCNNs) are mathematically equivalent to Tree Tensor Networks. They satisfy an "Area Law" of entanglement, meaning they cannot efficiently represent data where feature correlations are non-local (high Entanglement Entropy).
* **Quantum Link**: Quantum circuits operate in the full Hilbert space and are not constrained by the Area Law, allowing them to naturally represent "Volume Law" entanglement.


### 2.3 Micro-foundation: Neural Scaling Laws
**Source**: Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models.

* We define "Productivity" not as a black box, but via sample complexity:

$$\text{Error} \approx C N^{-\alpha}$$

* **Hypothesis**: The scaling exponent $\alpha$ for Classical models collapses as Data Entanglement increases.


## 3. Experimental Design (The "Lab")

### 3.1 The Agents (Model Pairs)

We pair "Local" Classical models with "Global" Quantum counterparts to isolate the locality mechanism. The key structural asymmetry in each pair is **how features are processed within the model's kernel/window**: classical models use local linear operations, while quantum models use entangled quantum circuits that process all features globally.

| **Domain** | **Classical Agent (Local Expert)** | **Quantum Counterpart (Global Expert)** | **Rationale for Comparison** |
| --- | --- | --- | --- |
| **Spatial** | **CNN** (ResNet-18) | **Quantum CNN (QCNN)** | Compares classical convolution vs. quantum subspace convolution. CNNs rely on $3\times3$ pixel locality. QCNNs process features globally via entangled quantum circuits within each kernel. |
| **Temporal** | **Classical TCN** | **Quantum TCN (QTCN)** | Both slide a local kernel window across the time axis, but differ in within-kernel processing. Classical TCN applies linear convolution filters. QTCN projects the window's features into qubits and applies a quantum circuit with U3, Ising gates, and entanglement-based pooling — capturing non-local correlations among features within each window. |
| **Attention** | **Local-Attention Transformer** | **Quantum Transformer (QTSTransformer)** | Compares windowed self-attention vs. quantum attention via QSVT. Local attention is blinded by global correlations beyond its window. QTSTransformer processes features through quantum state evolution in the full Hilbert space, naturally handling global feature correlations. |

#### 3.1.1 Why TCN replaces LSTM (change from v3)

The v3 plan paired Classical LSTM with Quantum LSTM. This pairing is problematic for the Feature Shuffling Protocol because **both share the same sequential recurrence structure**: at each time step $t$, both process $x_t$ and update $(h_t, c_t)$ identically in structure. Neither is "local" or "global" in the spatial sense that Alexander et al. describe — both see the full input vector at each step via their gate computations. Feature shuffling would affect both equally, producing no local-vs-global asymmetry.

The TCN/QTCN pairing resolves this by providing a clear structural analogy to CNN/QCNN:

| | **Image (Spatial)** | **Time Series (Temporal)** |
| --- | --- | --- |
| **Classical (Local)** | CNN scans with $3\times3$ kernel; linear filter within kernel | TCN scans with kernel_size window; linear conv within window |
| **Quantum (Global)** | QCNN uses quantum circuit within kernel | QTCN uses quantum circuit within kernel |
| **What shuffling breaks** | Pixel locality within CNN's receptive field | Feature locality within TCN's temporal window |


### 3.2 The Goods (Datasets & Perturbation)

We start with domains where Classical ML has an **Absolute Advantage** ($QE \approx 0$) and perturb them. Datasets span two domains (spatial and temporal), three task types (image classification, EEG classification, time-series regression), and a range of feature dimensionalities.

#### Spatial Domain (Image Classification)

| **Dataset** | **Features** | **Flat Dim** | **Task** | **Rationale** |
| --- | --- | --- | --- | --- |
| **MNIST** | $28 \times 28$ grayscale | 784 | 10-class classification | Sanity check; simple spatial structure; fast iteration |
| **CIFAR-10** | $32 \times 32 \times 3$ color | 3,072 | 10-class classification | Primary spatial benchmark; meaningful spatial locality in natural images |

#### Temporal Domain (Time-Series Classification & Regression)

For temporal data, each sample is flattened from $(C_{\text{channels}} \times T_{\text{timesteps}})$ into a 1D feature vector before shuffling. This destroys both temporal ordering and cross-channel spatial grouping simultaneously — directly analogous to pixel shuffling in images.

| **Dataset** | **Channels** | **Flat Dim** | **Task** | **Rationale** |
| --- | --- | --- | --- | --- |
| **PhysioNet EEG** | 64 electrodes | ~32,000 | Binary classification (left/right motor imagery) | High-dimensional; rich electrode topology provides strong spatial locality to destroy; standard QML benchmark |
| **ETTh1** | 7 variables | 672 ($7 \times 96$) | Regression (oil temperature) | Most-cited forecasting benchmark (Informer, PatchTST, iTransformer); strong diurnal (24h) and weekly (168h) periodicity |
| **Weather** | 21 meteorological variables | 2,016 ($21 \times 96$) | Regression (temperature) | Multi-scale periodicity (diurnal, seasonal); mix of periodic (temperature, radiation) and aperiodic (precipitation, wind) features; medium dimensionality |

#### The Independent Variable ($F_t$): Feature Shuffling Protocol

* We generate 11 variants of each dataset: $D_0, D_{10}, D_{20}, \dots, D_{100}$.
* $D_k$: A version where $k\%$ of feature indices are randomly permuted among themselves.
* The same permutation is applied to **all samples** in a dataset variant (column permutation, not row permutation).
* An isolated `torch.Generator` ensures the shuffle seed is independent of data loading randomness.
* _Effect_: Monotonically increases Quantum Entanglement ($QE$) without changing the information content (all features are preserved; only their ordering changes).

**Implementation**: The Feature Shuffling Protocol is implemented in `Load_Image_Datasets.py` via:
* `generate_feature_shuffle_permutation(n_features, shuffle_pct, seed)` — generates a partial permutation using an isolated RNG
* `apply_feature_shuffle(X, perm)` — applies the column permutation to all rows
* All dataset loaders accept `shuffle_pct` and `shuffle_seed` parameters
* `DatasetResult` dataclass returns loaders, raw CPU tensors (for QE computation), and the permutation vector


### 3.3 The Measures (Metrics)

**A. Independent Variable ($F_t$): Quantum Entanglement ($QE$)**
* Definition: The Von Neumann entropy of the singular values of the empirical data tensor.
* Method: Use **Algorithm 1** from Alexander et al. (2023) to compute this efficiently ($\mathcal{O}(M^3)$).
* Role: Proves that our perturbation actually increased physical complexity.

**B. Dependent Variable ($Y$): Test Performance**
* Classification tasks (MNIST, CIFAR-10, EEG): **Test Accuracy** and **ROC-AUC**.
* Regression tasks (ETTh1, Weather): **Test RMSE**.
* Role: The headline metric for "Productivity."

**C. Mechanism Diagnostics (The "Why")**
These metrics prove why the advantage occurs, satisfying theory reviewers.

1. **Effective Dimension ($d_{eff}$)** (Abbas et al., 2021):
    * Measures the "Capacity per Parameter." We expect Classical $d_{eff}$ to collapse on shuffled data.

2. **Spectral Signature / HTSR ($\alpha$)** (Martin & Mahoney, 2021):
    * Measures "Generalization." We expect Classical weights to degenerate to Random Matrix noise ($\alpha > 6$), while Quantum weights retain structure ($\alpha \approx 2$).

3. **Geometric Difference ($g_{diff}$)** (Huang et al., 2021):
    * Measures how well the model geometry matches the data geometry.


## 4. Methodology: Step-by-Step Protocol

### Phase 1: Data Preparation
1. Prepare all 5 datasets: MNIST, CIFAR-10, PhysioNet EEG, ETTh1, Weather.
2. Implement the **Feature Shuffling Protocol** to generate 11 variants ($D_0, D_{10}, \dots, D_{100}$) for each dataset. For temporal datasets, flatten $(C \times T)$ into a 1D vector before shuffling, then reshape back.
3. **Validation**: Run Alexander et al.'s Algorithm 1 on all variants. Plot $QE$ vs. Shuffling % for each dataset. Requirement: Must show monotonic increase across all 5 datasets.

### Phase 2: Model Training
1. Train all 6 Agents (3 Classical, 3 Quantum) on the appropriate datasets:

| **Model Pair** | **Datasets** | **Runs per seed** |
| --- | --- | --- |
| CNN vs. QCNN | MNIST, CIFAR-10 | $2 \times 2 \times 11 = 44$ |
| TCN vs. QTCN | EEG, ETTh1, Weather | $3 \times 2 \times 11 = 66$ |
| Local-Transformer vs. QTSTransformer | EEG, ETTh1, Weather | $3 \times 2 \times 11 = 66$ |
| **Total per seed** | | **176 training runs** |

2. Repeat with 3 seeds (2024, 2025, 2026) for statistical robustness. **Total: 528 training runs.**
3. Use standard hyperparameters (Adam optimizer, standard learning rates) to ensure fair comparison.
4. **Constraint**: Ensure Classical and Quantum models have comparable parameter counts or compute budgets to make the comparison fair.

### Phase 3: The Empirical Analysis (Interaction Regression)
Run the following regression to quantify Comparative Advantage:

$$\text{Performance}_{tdm} = \beta_0 + \beta_1 \text{Quantum}_m + \beta_2 (\text{Quantum}_m \times QE_t) + \gamma_d + \epsilon_{tdm}$$

where $\gamma_d$ is a dataset fixed effect.

* $\beta_1$ (Absolute Advantage): Expected $< 0$. (Quantum is worse at baseline).
* $\beta_2$ (Comparative Advantage): Expected $> 0$. (Quantum degrades slower than Classical).

Run the regression **separately for each model pair** (spatial, temporal, attention) and **pooled across all pairs** to test the generality of the result.

### Phase 4: Mechanism Analysis
1. Calculate **Generalization Gap** ($\text{Perf}_{train} - \text{Perf}_{test}$) for all runs. Plot vs. $QE$.
2. Calculate **Effective Dimension** ($d_{eff}$) using the Fisher Information Matrix for the trained models.
3. Calculate **HTSR Alpha** using the weightwatcher Python package.

### Phase 5: Counterfactual (Gains from Trade)
1. Define a "Manager" algorithm that routes tasks based on measured $QE$.
    * If $QE < \theta$: Use Classical Agent.
    * If $QE > \theta$: Use Quantum Agent.
2. Simulate the Total Loss of this Hybrid system vs. the "Autarky" (Classical only) system.
3. Report the % Efficiency Gain.


## 5. Expected Results & Contribution

| **Hypothesis** | **Metric** | **Expected Result** | **Significance** |
| --- | --- | --- | --- |
| **Absolute Disadvantage** | $\beta_1$ | Negative | Confirms Classical ML is superior for "natural" (low entanglement) data. |
| **Comparative Advantage** | $\beta_2$ | Positive | Proves Quantum ML is robust to entanglement; justifies specialization. |
| **Locality Failure** | $d_{eff}$ | Classical collapses | Confirms Alexander et al.'s theory that LCNNs lose capacity on entangled data. |
| **Cross-Domain Consistency** | $\beta_2 > 0$ across all 3 pairs | Positive in spatial, temporal, and attention | Shows the result is not architecture-specific but reflects a fundamental local-vs-global asymmetry. |
| **Economic Value** | Gains from Trade | $> 0\%$ | Provides the roadmap for commercial integration (Hybrid AI). |


## 6. Implementation Details

### 6.1 Codebase Structure

```
QComparativeAdvantage/
├── Load_Image_Datasets.py          # Spatial data loaders (MNIST, CIFAR) with feature shuffling
├── Load_PhysioNet_EEG.py           # EEG data loader
├── Load_TimeSeries_Datasets.py     # ETTh1, Weather data loaders (to be implemented)
├── EntropyCalc.py                  # QE computation (Alexander et al. Algorithm 1)
├── QCNN.py                         # Quantum CNN (spatial domain)
├── CNN_baseline.py                 # Classical CNN baseline (spatial domain)
├── QTCN.py                         # Quantum TCN (temporal domain)
├── TCN_baseline.py                 # Classical TCN baseline (temporal domain)
├── QTSTransformer.py               # Quantum Transformer (attention domain)
├── LocalTransformer_baseline.py    # Local-Attention Transformer baseline
├── run_experiment.py               # Experiment runner (sweep over shuffle levels)
└── analysis/
    ├── regression_analysis.py      # Interaction regression (Phase 3)
    ├── mechanism_analysis.py       # d_eff, HTSR, g_diff (Phase 4)
    └── gains_from_trade.py         # Hybrid routing simulation (Phase 5)
```

### 6.2 Feature Shuffling for Temporal Data

Temporal datasets require a flatten-shuffle-reshape pipeline:

```
Raw: (batch, channels, timesteps)
  → Flatten: (batch, channels × timesteps)
  → Shuffle: apply column permutation to the flattened vector
  → Reshape: (batch, channels, timesteps)   [for TCN/Transformer input]
```

The same permutation is applied to train, validation, and test splits. The raw (pre-device-transfer) CPU tensors are preserved in `DatasetResult` for QE computation.

### 6.3 SLURM Execution

* Account: `m4138_g`, Constraint: `gpu&hbm80g`, QOS: `shared`
* Classical runs: ~1-5 min/epoch (GPU), can batch multiple per job
* Quantum runs: ~60-90 min/epoch (quantum circuit simulation), one per job
* Total GPU-hours estimate: ~500-800 hours across all 528 runs


## 7. Key References

* Acemoglu, D., & Restrepo, P. (2018). The Race Between Man and Machine: Implications of Technology for Growth, Factor Shares, and Employment. _American Economic Review._
* Alexander, Y., et al. (2023). What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement. _NeurIPS._
* Abbas, A., et al. (2021). The power of quantum neural networks. _Nature Computational Science._
* Huang, H. Y., et al. (2021). Power of data in quantum machine learning. _Nature Communications._
* Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling laws for neural language models. _arXiv:2001.08361_
* Martin, C. H., & Mahoney, M. W. (2021). Implicit self-regularization in deep neural networks: Evidence from random matrix theory. _Journal of Machine Learning Research._
* Romalis, J. (2004). Factor Proportions, Trade, and Growth. _American Economic Review._
* Nunn, N. (2007). Relationship-Specificity, Incomplete Contracts, and the Pattern of Trade. _The Quarterly Journal of Economics._
* Costinot, A. (2009). On the Origins of Comparative Advantage. _Journal of International Economics._
* Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. _arXiv:1803.01271_
* Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. _AAAI 2021._
* Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. _NeurIPS 2021._
* Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. _ICLR 2023._
* Liu, Y., Hu, T., Zhang, H., et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. _ICLR 2024._


## Appendix A: Changes from v3

| **Aspect** | **v3** | **v4** | **Reason** |
| --- | --- | --- | --- |
| Temporal model pair | LSTM / QLSTM | TCN / QTCN | LSTM and QLSTM share the same recurrence structure; no local-vs-global asymmetry under feature shuffling. TCN/QTCN provides the same kernel-based asymmetry as CNN/QCNN. |
| Spatial datasets | CIFAR-10 only | MNIST + CIFAR-10 | MNIST as sanity check; two complexity levels strengthen the result. |
| Temporal datasets | Speech Commands | PhysioNet EEG, ETTh1, Weather | EEG provides high-dimensional multivariate classification (64 channels). ETTh1 and Weather are standard forecasting benchmarks cited by all major time-series papers (Informer, PatchTST, iTransformer). Three temporal datasets span biomedical, energy, and meteorological domains. |
| Temporal shuffling | Shuffle time steps | Flatten $(C \times T)$ → shuffle flat vector → reshape | Shuffling the joint channel-time vector destroys both temporal ordering and cross-channel correlations, directly analogous to pixel shuffling in images. |
| Regression model | $\text{Accuracy}_{tm}$ | $\text{Performance}_{tdm}$ with dataset fixed effects $\gamma_d$ | Accommodates multiple datasets and both classification (accuracy) and regression (RMSE) metrics within a single framework. |
| Experiment scale | 60 runs | 528 runs (176 per seed × 3 seeds) | More datasets, more model pairs, statistical robustness via multiple seeds. |
| Implementation | Not specified | Detailed codebase structure, `DatasetResult` API, SLURM estimates | Concrete implementation plan for reproducibility. |
