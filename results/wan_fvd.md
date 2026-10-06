Frechet distance to the baseline set in the baseline's PCA space (8 components, shrink 0.1); floor (mean over the perturbation controls) 0.002; sd = bootstrap over clips.

| configuration | n | speedup | distance | sd | excess over floor |
|---|---|---|---|---|---|
| perturb_1e-2 | 32 | 1.00 | 0.003 | 0.006 | control |
| perturb_1e-4 | 32 | 1.00 | 0.002 | 0.003 | control |
| cfg_trunc_0.6 | 32 | 1.22 | 0.001 | 0.000 | -0.002 |
| cfg0.4+pab_2 | 32 | 1.44 | 0.002 | 0.002 | +0.000 |
| wc_0.04 | 32 | 1.66 | 0.007 | 0.005 | +0.005 |
| ada_slow30 | 32 | 1.72 | 0.039 | 0.025 | +0.037 |
| fbc_0.1 | 32 | 1.73 | 0.022 | 0.018 | +0.019 |
| cfg0.6+wc_0.04 | 32 | 1.99 | 0.007 | 0.006 | +0.005 |
| wc_0.08 | 32 | 2.41 | 0.099 | 0.028 | +0.097 |
| ada_fast30 | 32 | 2.52 | 0.091 | 0.032 | +0.089 |
