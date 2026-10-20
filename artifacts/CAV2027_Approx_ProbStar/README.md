# CAV 2027 Artifact Evaluation: Approximate ProbStar & ProbSparseImageStar Reachability for Large Vision Models

This artifact implements **Approximate ProbStar Reachability** (`ProbSparseImageStar2DCSR`, `ProbSparseImageStar2DCOO`, and extended `ProbStar`) in **StarV** for quantitative probabilistic verification of large-scale vision models (CNNs, Semantic Segmentation Networks, ResNets, and Vision Transformers).

---

## 1. Key Contributions

1. **Single-Set Extended-Predicate Probabilistic Star Representation**:
   - Eliminates the $O(2^K)$ path-splitting explosion of Exact `ProbStar` across $K$ unstable neurons by introducing fresh relaxation predicate variables $\alpha_{\text{new}}$ in a single sparse 2D CSR/COO representation (`ProbSparseImageStar2DCSR` and `ProbSparseImageStar2DCOO`).
   - Supports both `cov_mode='diagonal'` ($O(n)$ memory and runtime for high-resolution vision models) and `cov_mode='full'` (dense covariance with Stein's lemma cross-covariance propagation).

2. **Dual Guarantee — Analytic Moment Matching + Provably Sound Probability Bounds**:
   - **Analytic Moment Propagation**: Closed-form Rectified-Gaussian moments for `ReLU` and `LeakyReLU`, Clark's pairwise Gaussian order-statistic approximation for `MaxPool2D`, and 5-point Gauss-Hermite quadrature + Stein's lemma for `GELU`, `LayerNorm`, and `MultiHeadAttention`.
   - **Provably Sound Linear Relaxation Envelope DAG (`RelaxationEnvelopeRecord`)**: Tracks exact linear lower and upper envelopes $(A_{\text{lb}}, b_{\text{lb}}, A_{\text{ub}}, b_{\text{ub}})$ across every nonlinear layer and back-substitutes output specifications to the original input Gaussian predicates $\alpha_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$, guaranteeing:
     $$0 \le p_{\min} \le p_{\text{true}} \le p_{\max} \le p_{\text{input}}, \qquad p_{\min} \le \hat{p} \le p_{\max}.$$

3. **Two-Stage Quantitative Specification Verifier (`verify_prob_spec`)**:
   - **Stage 1 (Polytope/LP Pruning)**: Uses fast range bounds and optional LP tightening on the Star polytope $C\alpha \le d$ to immediately certify `ROBUST` ($p_{\text{unsafe}} = 0$) or `VIOLATED` ($p_{\text{unsafe}} = p_{\text{input}}$) whenever qualitative reachability suffices.
   - **Stage 2 (Dual Back-Substitution + Moment Projection)**: When a specification is qualitatively `UNKNOWN`, projects both the back-substituted linear bounds (in $\alpha_0$-space) and the extended-predicate moments (in $\alpha$-space) onto the low-dimensional output specification subspace.

4. **Support Across Four Vision Architecture Families**:
   - **Classification CNNs**: `Conv2DLayer`, `BatchNorm2DLayer`, `ReLULayer`, `LeakyReLULayer`, `MaxPool2DLayer`, `AvgPool2DLayer`, `FlattenLayer`, `FullyConnectedLayer`.
   - **Semantic Segmentation Networks (SSNN / M2NIST)**: `ConvTranspose2DLayer` + `PixelClassificationLayer` with pixel-level probabilistic robustness maps (`certifyProbabilisticRobustness_pixel`).
   - **Residual Networks (ResNets)**: `ResidualLayer` with shared-predicate skip-connection addition (`residualAdd`) that avoids double-counting common ancestor uncertainty.
   - **Vision Transformers (ViTs)**: `LayerNormLayer`, `MultiHeadAttentionLayer` (bilinear McCormick QK$^T$ + monotone Softmax enclosure + value-path propagation), and `GELULayer`.

---

## 2. Comparison with Qualitative & Quantitative Verifiers

| Verifier / Method | Category | Vision Architectures | Guarantee Type | Output | Path Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Star / ImageStar** | Qualitative | FFNN, CNN, SSNN | Sound (Exact / Over-Approx) | Yes / No / Unknown | $O(2^K)$ Exact, $O(1)$ Approx |
| **SparseImageStar (CSR/COO)** | Qualitative | CNN, SSNN, VGG | Sound Over-Approx | Yes / No / Unknown | $O(1)$ set |
| **DeepPoly / ERAN** | Qualitative | FFNN, CNN, ResNet | Sound Single-Neuron | Yes / No / Unknown | $O(1)$ bounds |
| **$\alpha,\beta$-CROWN** | Qualitative | FFNN, CNN, ResNet, ViT | Sound & Complete (BaB) | Yes / No / Unknown | $O(2^B)$ BaB |
| **Exact ProbStar** | Quantitative | Small FFNN / Micro-CNN | Exact Probability | Exact $p_{\text{safe}}$ | $O(2^K)$ sets |
| **Filtered ProbStar** | Quantitative | Small FFNN / Micro-CNN | Sound Bounds $[p_{\text{lb}}, p_{\text{ub}}]$ | Bounded interval | $O(2^{K_{\text{eff}}})$ sets |
| **Monte Carlo (Clopper-Pearson)** | Quantitative | Black-box | Statistical $(1-\delta)$ CI | Empirical $\hat{p}_{\text{MC}}$ | $O(N_{\text{samples}})$ |
| **ProbSparseImageStar (CSR/COO) [Ours]** | **Quantitative + Qualitative** | **CNN, VGG, SSNN, ResNet, ViT** | **Sound $[p_{\min}, p_{\max}]$ + Moment $\hat{p}$** | **Exact Status + Probability Bounds & Heatmaps** | **$O(1)$ set** |

---

## 3. Reproducing the Unit Tests & CAV 2027 Benchmark Results

### Run Unit & Comparative Soundness Tests
```bash
python -m unittest tests/test_approx_probstar_vision.py
```

### Run Full CAV 2027 Benchmark Suite
```bash
python -m artifacts.CAV2027_Approx_ProbStar.CAV2027_Approx_ProbStar
```

### Generated Artifacts (`artifacts/CAV2027_Approx_ProbStar/results/`)
- `suite1_exact_vs_approx.csv` — Exact ProbStar vs. Filtered ProbStar vs. Approx ProbStar (Analytic & Heuristic) vs. Qualitative Star vs. Monte Carlo.
- `suite2_cnn_benchmarks.csv` — Qualitative (`ImageStar`, `SIM_CSR`, `SIM_COO`) vs. Quantitative (`PSIM_CSR`, `PSIM_COO`, `MC`) across MNIST ConvNet, CIFAR-10 OVAL21 CNN, and VGG-Stage CNN.
- `suite3_ssnn_benchmarks.csv` — Semantic Segmentation (`M2NIST_SSNN`) qualitative vs. quantitative pixel certification.
- `suite4_resnet_vit_benchmarks.csv` — ResNet (`ResidualLayer`) and Vision Transformer (`LayerNormLayer`, `MultiHeadAttentionLayer`, `GELULayer`) probabilistic certification and pointwise envelope verification.
- `verifier_taxonomy.csv` & `cav2027_tables.tex` — Publication-ready LaTeX tables for CAV 2027.
- `fig1_runtime_memory_scaling.png` & `fig2_ssnn_prob_heatmaps.png` — Figures comparing runtime/memory scaling and pixel-wise probabilistic robustness heatmaps.
