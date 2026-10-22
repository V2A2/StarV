# CAV 2027 Artifact: Approximate ProbStar Reachability for Verifying Large Vision Models

This artifact implements and evaluates **Approximate ProbStar** (`ProbStar`) and **Probabilistic Sparse ImageStar** (`ProbSparseImageStar2DCSR` and `ProbSparseImageStar2DCOO`) for scalable quantitative & qualitative verification of **real full-size pretrained vision models** (CNNs, Semantic Segmentation Networks, ResNets, and Vision Transformers).

## 1. Overview of New Modules & Pretrained Models

### Core Set Representations & Layer Relaxations
- [`StarV/set/probsparse_base.py`](../../StarV/set/probsparse_base.py): Core base class `ProbSparseBase` implementing two-sided linear envelope DAG (`env_dag`) tracking, exact dual-projected back-substitution (`getSoundInputBounds`, `getSoundInputMarginBounds`), analytical moment propagation (`mu`, `Sig`), and Gaussian-correlated (`mvncdf`) / Chebyshev probability bounds (`[p_min, p_max]`, `p_hat`).
- [`StarV/set/probsparseimagestar2dcsr.py`](../../StarV/set/probsparseimagestar2dcsr.py) & [`StarV/set/probsparseimagestar2dcoo.py`](../../StarV/set/probsparseimagestar2dcoo.py): Sparse 2D CSR and COO probabilistic image star representations.
- [`StarV/layer/ResidualLayer.py`](../../StarV/layer/ResidualLayer.py): DAG skip-connection layer ($y = \phi(F(x) + G(x))$) fusing shared predicate generators across main and shortcut branches.
- [`StarV/layer/LeakyReLULayer.py`](../../StarV/layer/LeakyReLULayer.py): Exact and single-set triangle relaxation for LeakyReLU with exact Gaussian truncated-normal moment propagation.
- [`StarV/layer/GELULayer.py`](../../StarV/layer/GELULayer.py): Analytical tangent/chord linear envelope relaxation for GELU.
- [`StarV/layer/LayerNormLayer.py`](../../StarV/layer/LayerNormLayer.py): Coupled numerator-denominator token-wise LayerNorm relaxation with Cauchy-Schwarz leave-one-out variance lower bounds.
- [`StarV/layer/MultiHeadAttentionLayer.py`](../../StarV/layer/MultiHeadAttentionLayer.py): Bilinear Taylor-McCormick attention score bounds with affine score-difference cancellation and output-projected box-simplex water-filling for Vision Transformers (ViTs).
- [`StarV/verifier/verifier.py`](../../StarV/verifier/verifier.py) & [`StarV/verifier/certifier.py`](../../StarV/verifier/certifier.py): Single-set quantitative BFS reachability (`quantiVerifyApproxBFS`), Monte Carlo baseline (`quantiVerifyVisionMC`), image classification certification (`certifyProbabilisticRobustness`), and pixel-wise semantic segmentation certification (`certifyProbabilisticRobustness_pixel`).

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

# Run full CAV 2027 benchmark suite (Suites 1-4) and generate CSV/LaTeX/JSON/PNG outputs
python -m artifacts.CAV2027_Approx_ProbStar.CAV2027_Approx_ProbStar --suite all
```

## 3. Summary of Experimental Results on Real Full-Size Models

### Suite 1: Exact ProbStar vs. Filtered vs. Approx ProbStar vs. Qualitative Star vs. MC
| $\epsilon$ | Qual Star | `#Exact` | `#Approx` | `[p_min, p_max]` | `p_hat` | `p_exact` | `p_MC` | `t_Exact` | `t_Approx` | Speedup |
|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.05 | ROBUST | 2 | 1 | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 1.0000 | 3.1317s | 0.0006s | **5,198.9x** |
| 0.15 | ROBUST | 7 | 1 | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 1.0000 | 7.3285s | 0.0005s | **16,106.5x** |
| 0.28 | UNKNOWN | 11 | 1 | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 1.0000 | 14.3771s | 0.0006s | **22,215.5x** |
| 0.42 | UNKNOWN | 30 | 1 | `[0.9990, 1.0000]` | 1.0000 | 1.0000 | 1.0000 | 27.7498s | 0.0007s | **41,322.6x** |
| 0.65 | UNKNOWN | 91 | 1 | `[0.9056, 1.0000]` | 0.9995 | 1.0000 | 1.0000 | 20.9777s | 0.0007s | **31,899.1x** |

### Suite 2: Full-Size Pretrained CNNs ($28 \times 28 \times 1$ CAV2020 MNIST & $32 \times 32 \times 3$ OVAL21 CIFAR-10)
| Model | Input Shape | Attack Spec | `#Pred` | Qual(CSR) | `[p_min, p_max]` | `p_hat` | `p_MC` | `t_NNV` | `t_SIM_CSR` | `t_PSIM_CSR` | `Mem(SIM)` | `Mem(PSIM)` |
|:---|:---|:---|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| `CAV20_MNIST_Small` | `28x28x1` | `bright(d=250,dlt=0.005)` | 16 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.21s | 0.0505s | **0.0359s** | 0.00MB | 0.00MB |
| `CAV20_MNIST_Small` | `28x28x1` | `bright(d=240,dlt=0.015)` | 96 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.99s | 0.0478s | **0.0513s** | 0.03MB | 0.04MB |
| `CAV20_MNIST_Medium` | `28x28x1` | `bright(d=250,dlt=0.005)` | 19 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.43s | 0.0732s | **0.0806s** | 0.00MB | 0.01MB |
| `CAV20_MNIST_Medium` | `28x28x1` | `bright(d=240,dlt=0.015)` | 139 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 5.40s | 0.1112s | **0.1126s** | 0.10MB | 0.14MB |
| `CAV20_MNIST_Large` | `28x28x1` | `bright(d=250,dlt=0.005)` | 31 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.55s | 0.1524s | **0.1621s** | 0.01MB | 0.01MB |
| `CAV20_MNIST_Large` | `28x28x1` | `bright(d=240,dlt=0.015)` | 187 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 5.45s | 0.3639s | **0.4001s** | 0.18MB | 0.27MB |
| `OVAL21_CIFAR_Base` | `32x32x3` | `patch_8x8x3(eps=0.05)` | 196 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | N/A | 0.0135s | **0.0103s** | 0.03MB | 0.04MB |
| `OVAL21_CIFAR_Base` | `32x32x3` | `full_32x32x3(eps=0.005)` | 3,141 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | N/A | 0.0698s | **0.0567s** | 0.84MB | 1.28MB |
| `OVAL21_CIFAR_Base` | `32x32x3` | `full_32x32x3(eps=0.03)` | 3,180 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | N/A | 0.0632s | **0.0640s** | 1.00MB | 1.53MB |
| `OVAL21_CIFAR_Deep` | `32x32x3` | `patch_8x8x3(eps=0.05)` | 209 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | N/A | 0.0166s | **0.0166s** | 0.08MB | 0.12MB |
| `OVAL21_CIFAR_Deep` | `32x32x3` | `full_32x32x3(eps=0.005)` | 3,198 | UNKNOWN | **`[1.0000, 1.0000]`** | 1.0000 | 1.0000 | 23.88s | 0.1711s | **0.2435s** | 1.39MB | 2.12MB |
| `OVAL21_CIFAR_Deep` | `32x32x3` | `full_32x32x3(eps=0.03)` | 3,252 | UNKNOWN | **`[1.0000, 1.0000]`** | 1.0000 | 1.0000 | 23.88s | 0.2145s | **0.1879s** | 1.91MB | 2.90MB |
| `OVAL21_CIFAR_Wide` | `32x32x3` | `patch_8x8x3(eps=0.05)` | 204 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | N/A | 0.0252s | **0.0212s** | 0.06MB | 0.08MB |
| `OVAL21_CIFAR_Wide` | `32x32x3` | `full_32x32x3(eps=0.005)` | 3,230 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | N/A | 0.1545s | **0.1250s** | 1.08MB | 1.65MB |
| `OVAL21_CIFAR_Wide` | `32x32x3` | `full_32x32x3(eps=0.03)` | 3,296 | UNKNOWN | **`[1.0000, 1.0000]`** | 1.0000 | 1.0000 | N/A | 0.1797s | **0.1313s** | 1.37MB | 2.08MB |

### Suite 3: Full-Size Pretrained Semantic Segmentation ($28 \times 28 \times 1$ & $28 \times 56 \times 1$ M2NIST)
| SSNN Model | Input Shape | `AttPx` | $\epsilon$ | `#Pred` | Qual Rob% | Qual Unk% | `E[p_min]` | `E[p_hat]` | `t_SIM_CSR` | `t_PSIM_CSR` | `Mem(MB)` |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M2NIST_SSNN_28x28` | `28x28x1` | 64 | 0.03 | 91 | 95.5% | 4.5% | **0.9987** | **0.9991** | 0.0503s | 0.0633s | 1.43MB |
| `M2NIST_SSNN_28x28` | `28x28x1` | 196 | 0.08 | 876 | 66.3% | 33.7% | **0.8959** | **0.9875** | 0.0693s | 0.6166s | 9.29MB |
| `M2NIST_SSNN_28x56` | `28x56x1` | 128 | 0.03 | 197 | 94.6% | 5.4% | **0.9971** | **0.9984** | 0.0841s | 0.1832s | 3.24MB |
| `M2NIST_SSNN_28x56` | `28x56x1` | 392 | 0.08 | 1,829 | 64.3% | 35.7% | **0.8814** | **0.9842** | 0.1542s | 2.1404s | 20.19MB |

### Suite 4: Full-Size Pretrained ResNet & Vision Transformer ($28 \times 28 \times 1$ & $32 \times 32 \times 3$)
| ResNet/ViT Arch | Input Shape | `AttPx` | $\epsilon$ | `#Pred` | Qual(CSR) | `[p_min, p_max]` | `p_hat` | `p_MC` | `t_SIM_CSR` | `t_PSIM_CSR` | `Mem(MB)` | `EnvViol` |
|:---|:---|---:|---:|---:|:---|:---|---:|---:|---:|---:|---:|:---|
| `ResNet_MNIST_28x28` | `28x28x1` | 64 | 0.008 | 91 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.0215s | **0.0220s** | 0.06MB | `0/100` |
| `ResNet_MNIST_28x28` | `28x28x1` | 64 | 0.030 | 227 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.0355s | **0.0276s** | 0.45MB | `0/100` |
| `ResNet_CIFAR10_32x32` | `32x32x3` | 192 | 0.008 | 221 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.0411s | **0.0444s** | 0.16MB | `0/100` |
| `ResNet_CIFAR10_32x32` | `32x32x3` | 192 | 0.030 | 358 | UNKNOWN | **`[1.0000, 1.0000]`** | 1.0000 | 1.0000 | 0.0601s | **0.0496s** | 1.07MB | `0/100` |
| `ViT_MNIST_28x28` | `28x28x1` | 49 | 0.008 | 683 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.0464s | **0.0579s** | 0.55MB | `0/100` |
| `ViT_MNIST_28x28` | `28x28x1` | 49 | 0.030 | 719 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.0576s | **0.0543s** | 0.58MB | `0/100` |
| `ViT_CIFAR10_32x32` | `32x32x3` | 192 | 0.008 | 1,232 | ROBUST | `[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.0597s | **0.0620s** | 1.01MB | `0/100` |
| `ViT_CIFAR10_32x32` | `32x32x3` | 192 | 0.030 | 1,232 | UNKNOWN | `[0.0000, 1.0000]` | 1.0000 | 1.0000 | 0.0657s | **0.0664s** | 1.01MB | `0/100` |
