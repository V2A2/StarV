# CAV 2027 Artifact: Approximate ProbStar Reachability for Verifying Large Vision Models

This artifact implements and evaluates **Approximate ProbStar** (`ProbStar`) and **Probabilistic Sparse ImageStar** (`ProbSparseImageStar2DCSR` and `ProbSparseImageStar2DCOO`) for scalable quantitative & qualitative verification of **real full-size pretrained vision models** (CNNs, Semantic Segmentation Networks, ResNets, and Vision Transformers), and compares them against **CROWN**, **$\alpha$-CROWN**, **$\alpha,\beta$-CROWN**, **NNV (`ImageStar`)**, **SparseImageStar (`SIM_CSR`/`SIM_COO`)**, **Exact/Filtered `ProbStar`**, and **Monte Carlo (`MC`)**.

## 1. Overview of Modules & Pretrained Models

### Core Set Representations, Layer Relaxations, and Verifiers
- [`StarV/set/probsparse_base.py`](../../StarV/set/probsparse_base.py): Core base class `ProbSparseBase` implementing two-sided linear envelope DAG (`env_dag`) tracking, exact dual-projected back-substitution (`getSoundInputBounds`, `getSoundInputMarginBounds`), analytical moment propagation (`mu`, `Sig`), and Gaussian-correlated (`mvncdf`) / Chebyshev probability bounds (`[p_min, p_max]`, `p_hat`).
- [`StarV/set/probsparseimagestar2dcsr.py`](../../StarV/set/probsparseimagestar2dcsr.py) & [`StarV/set/probsparseimagestar2dcoo.py`](../../StarV/set/probsparseimagestar2dcoo.py): Sparse 2D CSR and COO probabilistic image star representations.
- [`StarV/verifier/crown.py`](../../StarV/verifier/crown.py): Complete backward Linear Relaxation-based Perturbation Analysis (`CROWN`, `alpha-CROWN`, and `alpha,beta-CROWN` with Branch-and-Bound) for both qualitative (`qualiVerifyCROWN`, `certifyRobustness_CROWN`) and quantitative (`quantiVerifyCROWN`, `certifyProbabilisticRobustness_CROWN`) verification across FNN, CNN, SSNN, ResNet, and ViT architectures.
- [`StarV/layer/ResidualLayer.py`](../../StarV/layer/ResidualLayer.py): DAG skip-connection layer ($y = \phi(F(x) + G(x))$) fusing shared predicate generators across main and shortcut branches.
- [`StarV/layer/LeakyReLULayer.py`](../../StarV/layer/LeakyReLULayer.py): Exact and single-set triangle relaxation for LeakyReLU with exact Gaussian truncated-normal moment propagation.
- [`StarV/layer/GELULayer.py`](../../StarV/layer/GELULayer.py): Analytical tangent/chord linear envelope relaxation for GELU.
- [`StarV/layer/LayerNormLayer.py`](../../StarV/layer/LayerNormLayer.py): Coupled numerator-denominator token-wise LayerNorm relaxation with Cauchy-Schwarz leave-one-out variance lower bounds.
- [`StarV/layer/MultiHeadAttentionLayer.py`](../../StarV/layer/MultiHeadAttentionLayer.py): Bilinear Taylor-McCormick attention score bounds with affine score-difference cancellation and output-projected box-simplex water-filling for Vision Transformers (ViTs).
- [`StarV/verifier/verifier.py`](../../StarV/verifier/verifier.py) & [`StarV/verifier/certifier.py`](../../StarV/verifier/certifier.py): Single-set quantitative BFS reachability (`quantiVerifyApproxBFS`), Monte Carlo baseline (`quantiVerifyVisionMC`), image classification certification (`certifyProbabilisticRobustness`), and pixel-wise semantic segmentation certification (`certifyProbabilisticRobustness_pixel`, supporting both `reachMethod='approx'` and `reachMethod='crown'`).

### Pretrained Full-Size Vision Benchmarks
1. **CAV 2020 Pretrained MNIST ConvNets ($28 \times 28 \times 1$)**: `Small_ConvNet.onnx` (14L), `Medium_ConvNet.onnx` (20L), `Large_ConvNet.onnx` (26L) in `StarV/util/data/nets/CAV2020_MNIST_ConvNet/onnx/` with NNV baselines.
2. **OVAL21 Pretrained CIFAR-10 CNNs ($32 \times 32 \times 3$)**: `cifar_base_kw.onnx` (8L), `cifar_deep_kw.onnx` (12L), `cifar_wide_kw.onnx` (8L) in `StarV/util/data/nets/oval21/onnx/` with all 3,072 input pixels perturbed (`oval21_memory.mat`) and NNV baselines (`nnv_oval21_memory_usage.mat`).
3. **Full-Size Pretrained M2NIST SSNNs, ResNets, and ViTs**: Stored in [`StarV/util/data/nets/CAV2027_Vision/pretrained_models.npz`](../../StarV/util/data/nets/CAV2027_Vision/pretrained_models.npz):
   - `M2NIST_SSNN_28x28` ($28 \times 28 \times 1 \to 28 \times 28 \times 11$ classes, 89.78% pixel accuracy) & `M2NIST_SSNN_28x56` ($28 \times 56 \times 1 \to 28 \times 56 \times 11$ classes, two-digit segmentation, 17,248 output logits).
   - `ResNet_MNIST_28x28` ($28 \times 28 \times 1$, 92.80% test accuracy) & `ResNet_CIFAR10_32x32` ($32 \times 32 \times 3$).
   - `ViT_MNIST_28x28` ($28 \times 28 \times 1$, $7 \times 7$ patches $\to 16 \times 16$ tokens, 4 heads, 95.05% test accuracy) & `ViT_CIFAR10_32x32` ($32 \times 32 \times 3$, $8 \times 8$ patches $\to 16 \times 16$ tokens, 4 heads).

## 2. Reproducing All Experiments & Unit Tests

```bash
# Run unit tests
python -m tests.test_approx_probstar_vision
python -m tests.test_crown

# Run full CAV 2027 benchmark suite (Suites 1-4) and generate CSV/LaTeX/JSON/PNG outputs
python -m artifacts.CAV2027_Approx_ProbStar.CAV2027_Approx_ProbStar --suite all
```

## 3. Summary of Experimental Results & Comparison with CROWN

### Suite 1: Exact ProbStar vs. Filtered vs. Approx ProbStar vs. CROWN / $\alpha$-CROWN / $\alpha,\beta$-CROWN vs. MC
| $\epsilon$ | Qual (Star / CROWN / $\alpha\beta$-CROWN) | `#Ex` | `#Ap` | Ours `[p_min, p_max]` | Ours `p_hat` | CROWN `[p_min, p_max]` | $\alpha\beta$-CROWN `[p_min, p_max]` | `p_exact` | `p_MC` | `t_Exact` | `t_Ours` | `t_CROWN` | `t_abCR` | Speedup vs Exact |
|---:|:---|---:|---:|:---|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | `ROBUST/ROBUST/ROBUST` | 2 | 1 | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 3.1731s | 0.0008s | 0.0004s | 0.0118s | **4,044.4x** |
| 0.15 | `ROBUST/ROBUST/ROBUST` | 7 | 1 | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 7.2809s | 0.0005s | 0.0004s | 0.0215s | **15,387.6x** |
| 0.28 | `UNKNOWN/ROBUST/ROBUST` | 11 | 1 | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 13.8575s | 0.0007s | 0.0003s | 0.0204s | **18,733.7x** |
| 0.42 | `UNKNOWN/UNKNOWN/ROBUST` | 30 | 1 | **`[0.9990, 1.0000]`** | 1.0000 | `[0.9899, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 26.5549s | 0.0007s | 0.0005s | 0.0204s | **39,128.1x** |
| 0.65 | `UNKNOWN/UNKNOWN/UNKNOWN` | 91 | 1 | `[0.9056, 1.0000]` | **0.9995** | `[0.9168, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 19.8530s | 0.0006s | 0.0005s | 0.1079s | **33,410.9x** |

### Suite 2: Full-Size Pretrained CNNs ($28 \times 28 \times 1$ CAV2020 MNIST & $32 \times 32 \times 3$ OVAL21 CIFAR-10) vs. CROWN & NNV
| Model | Input Shape | Attack Spec | `#Pred` | Qual (CSR / CROWN) | Ours `[p_min, p_max]` | Ours `p_hat` | CROWN `[p_min, p_max]` | `p_MC` | `t_NNV` | `t_SIM_CSR` | `t_CROWN` | `t_PSIM_CSR` | Speedup vs CROWN | `Mem(PSIM)` |
|:---|:---|:---|---:|:---|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| `CAV20_MNIST_Small` | `28x28x1` | `bright(d=250,dlt=0.005)` | 16 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | 0.21s | 0.0466s | 2.5400s | **0.0422s** | **60.1x** | 0.00MB |
| `CAV20_MNIST_Small` | `28x28x1` | `bright(d=240,dlt=0.015)` | 96 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | 0.99s | 0.0329s | 2.5657s | **0.0333s** | **76.9x** | 0.04MB |
| `CAV20_MNIST_Medium` | `28x28x1` | `bright(d=250,dlt=0.005)` | 19 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | 0.43s | 0.0706s | 5.1800s | **0.0679s** | **76.3x** | 0.01MB |
| `CAV20_MNIST_Medium` | `28x28x1` | `bright(d=240,dlt=0.015)` | 139 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | 5.40s | 0.1236s | 5.2139s | **0.1051s** | **49.6x** | 0.14MB |
| `CAV20_MNIST_Large` | `28x28x1` | `bright(d=250,dlt=0.005)` | 31 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | 0.55s | 0.2131s | 17.6594s | **0.2079s** | **84.9x** | 0.01MB |
| `CAV20_MNIST_Large` | `28x28x1` | `bright(d=240,dlt=0.015)` | 187 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | 5.45s | 0.3073s | 19.0174s | **0.3106s** | **61.2x** | 0.27MB |
| `OVAL21_CIFAR_Base` | `32x32x3` | `patch_8x8x3(eps=0.05)` | 196 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | N/A | 0.0099s | 0.5374s | **0.0098s** | **54.7x** | 0.04MB |
| `OVAL21_CIFAR_Base` | `32x32x3` | `full_32x32x3(eps=0.005)` | 3,141 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | N/A | 0.0680s | 1.1379s | **0.0673s** | **16.9x** | 1.28MB |
| `OVAL21_CIFAR_Base` | `32x32x3` | `full_32x32x3(eps=0.03)` | 3,180 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | N/A | 0.0568s | 1.1514s | **0.0882s** | **13.1x** | 1.53MB |
| `OVAL21_CIFAR_Deep` | `32x32x3` | `patch_8x8x3(eps=0.05)` | 209 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | N/A | 0.0265s | 1.7549s | **0.0310s** | **56.5x** | 0.12MB |
| `OVAL21_CIFAR_Deep` | `32x32x3` | `full_32x32x3(eps=0.005)` | 3,198 | `UNKNOWN/UNKNOWN` | **`[1.0000, 1.0000]`** | 1.0000 | **`[1.0000, 1.0000]`** | 1.0000 | 23.88s | 0.1971s | 3.1044s | **0.2165s** | **14.3x** | 2.12MB |
| `OVAL21_CIFAR_Deep` | `32x32x3` | `full_32x32x3(eps=0.03)` | 3,252 | `UNKNOWN/UNKNOWN` | **`[1.0000, 1.0000]`** | 1.0000 | **`[1.0000, 1.0000]`** | 1.0000 | 23.88s | 0.2057s | 3.0188s | **0.1985s** | **15.2x** | 2.90MB |
| `OVAL21_CIFAR_Wide` | `32x32x3` | `patch_8x8x3(eps=0.05)` | 204 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | N/A | 0.0286s | 2.0262s | **0.0444s** | **45.6x** | 0.08MB |
| `OVAL21_CIFAR_Wide` | `32x32x3` | `full_32x32x3(eps=0.005)` | 3,230 | `ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | N/A | 0.1747s | 2.9409s | **0.1742s** | **16.9x** | 1.65MB |
| `OVAL21_CIFAR_Wide` | `32x32x3` | `full_32x32x3(eps=0.03)` | 3,296 | `UNKNOWN/ROBUST` | **`[1.0000, 1.0000]`** | 1.0000 | `[1.0000, 1.0000]` | 1.0000 | N/A | 0.1609s | 3.0387s | **0.1678s** | **18.1x** | 2.08MB |

### Suite 3: Full-Size Pretrained Semantic Segmentation ($28 \times 28 \times 1$ & $28 \times 56 \times 1$ M2NIST) vs. CROWN
| SSNN Model | Input Shape | `AttPx` | $\epsilon$ | `#Pred` | `Rob%(CSR)` | `Rob%(CROWN)` | Ours `E[p_min]` | Ours `E[p_hat]` | CROWN `E[p_min]` | CROWN `E[p_hat]` | `t_SIM_CSR` | `t_CROWN` | `t_PSIM_CSR` | Speedup vs CROWN | `Mem(MB)` |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M2NIST_SSNN_28x28` | `28x28x1` | 64 | 0.03 | 91 | 95.5% | 96.3% | **0.9987** | **0.9991** | 0.9987 | 0.9994 | 0.0457s | 8.0955s | **0.0683s** | **118.5x** | 1.43MB |
| `M2NIST_SSNN_28x28` | `28x28x1` | 196 | 0.08 | 876 | 66.3% | 83.5% | **0.8959** | **0.9875** | 0.9056 | 0.9832 | 0.0923s | 10.0471s | **0.6123s** | **16.4x** | 9.29MB |
| `M2NIST_SSNN_28x56` | `28x56x1` | 128 | 0.03 | 197 | 94.6% | 95.1% | **0.9971** | **0.9984** | 0.9971 | 0.9985 | 0.1099s | 30.9668s | **0.1770s** | **175.0x** | 3.24MB |
| `M2NIST_SSNN_28x56` | `28x56x1` | 392 | 0.08 | 1,829 | 64.3% | 81.8% | **0.8814** | **0.9842** | 0.8917 | 0.9758 | 0.2216s | 34.5471s | **2.4094s** | **14.3x** | 20.19MB |

### Suite 4: Full-Size Pretrained ResNet & Vision Transformer ($28 \times 28 \times 1$ & $32 \times 32 \times 3$) vs. CROWN / $\alpha$-CROWN
| ResNet/ViT Arch | Input Shape | `AttPx` | $\epsilon$ | `#Pred` | Qual (CSR / CROWN / $\alpha$-CROWN) | Ours `[p_min, p_max]` | Ours `p_hat` | CROWN `[p_min, p_max]` | $\alpha$-CROWN `[p_min, p_max]` | `p_MC` | `t_SIM_CSR` | `t_CROWN` | `t_aCROWN` | `t_PSIM_CSR` | Speedup vs CROWN | `Mem(MB)` | `EnvViol` |
|:---|:---|---:|---:|---:|:---|:---|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| `ResNet_MNIST_28x28` | `28x28x1` | 64 | 0.008 | 91 | `ROBUST/ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0258s | 1.2246s | 16.3951s | **0.0240s** | **51.1x** | 0.06MB | `0/100` |
| `ResNet_MNIST_28x28` | `28x28x1` | 64 | 0.030 | 227 | `ROBUST/ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0294s | 1.4422s | 17.1219s | **0.0282s** | **51.2x** | 0.45MB | `0/100` |
| `ResNet_CIFAR10_32x32` | `32x32x3` | 192 | 0.008 | 221 | `ROBUST/ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0384s | 2.3851s | 28.3618s | **0.0377s** | **63.3x** | 0.16MB | `0/100` |
| `ResNet_CIFAR10_32x32` | `32x32x3` | 192 | 0.030 | 358 | `UNKNOWN/ROBUST/ROBUST` | **`[1.0000, 1.0000]`** | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0443s | 2.0714s | 29.1336s | **0.0417s** | **49.7x** | 1.07MB | `0/100` |
| `ViT_MNIST_28x28` | `28x28x1` | 49 | 0.008 | 683 | `ROBUST/ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0831s | 0.1743s | 0.2948s | **0.0647s** | **2.7x** | 0.55MB | `0/100` |
| `ViT_MNIST_28x28` | `28x28x1` | 49 | 0.030 | 719 | `ROBUST/ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0735s | 0.1680s | 0.3150s | **0.0666s** | **2.5x** | 0.58MB | `0/100` |
| `ViT_CIFAR10_32x32` | `32x32x3` | 192 | 0.008 | 1,232 | `ROBUST/ROBUST/ROBUST` | `[1.0000, 1.0000]` | 1.0000 | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0766s | 0.2445s | 0.5485s | **0.0905s** | **2.7x** | 1.01MB | `0/100` |
| `ViT_CIFAR10_32x32` | `32x32x3` | 192 | 0.030 | 1,232 | `UNKNOWN/ROBUST/ROBUST` | `[0.0000, 1.0000]` | **1.0000** | `[1.0000, 1.0000]` | `[1.0000, 1.0000]` | 1.0000 | 0.0785s | 0.2703s | 0.5637s | **0.0785s** | **3.4x** | 1.01MB | `0/100` |
