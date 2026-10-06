Policy returns on the Dreamer walker (8 real episodes per policy; imagined over 60 steps from 24 held-out start states).

| policy | true return | baseline | baseline_reseed | gumbel | tensorrt_fp16 | latent_noise_0.5 | latent_noise_0 | bf16_autocast | int8_weight_only | low_rank_0.5 | low_rank_0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| actor | 824 | 111.7 | 111.8 | 111.7 | 112.6 | 105.6 | 99.0 | 111.8 | 113.2 | 100.1 | 40.2 |
| noise_0.1 | 801 | 110.3 | 109.3 | 110.3 | 110.7 | 106.2 | 92.2 | 109.3 | 110.9 | 96.2 | 36.6 |
| noise_0.3 | 663 | 104.2 | 107.7 | 104.2 | 104.8 | 102.5 | 88.5 | 108.9 | 103.7 | 84.4 | 35.0 |
| noise_0.2 | 644 | 109.8 | 109.3 | 109.8 | 109.3 | 104.0 | 89.1 | 109.0 | 105.7 | 90.9 | 35.0 |
| noise_0.4 | 641 | 106.7 | 104.1 | 106.7 | 107.4 | 97.3 | 88.3 | 108.4 | 105.2 | 75.9 | 34.8 |
| noise_0.5 | 524 | 99.7 | 96.1 | 99.7 | 98.6 | 92.6 | 82.1 | 95.9 | 95.9 | 70.7 | 30.6 |
| noise_0.6 | 442 | 89.2 | 94.7 | 89.2 | 91.8 | 83.8 | 77.1 | 95.0 | 91.7 | 59.5 | 29.2 |
| noise_0.8 | 350 | 64.7 | 68.6 | 64.7 | 65.3 | 66.8 | 56.3 | 71.0 | 68.9 | 46.6 | 24.9 |
| noise_1 | 308 | 49.1 | 51.8 | 49.1 | 51.4 | 53.3 | 49.1 | 53.4 | 52.4 | 39.7 | 25.7 |

Agreement of each model's ranking with the real simulator and with the unoptimized model's:

| model | vs simulator | vs unoptimized model |
|---|---|---|
| baseline | spearman +0.95; kendall +0.89; pairwise 0.94; same best policy yes (n=9) |  |
| baseline_reseed | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) |
| gumbel | spearman +0.95; kendall +0.89; pairwise 0.94; same best policy yes (n=9) | spearman +1.00; kendall +1.00; pairwise 1.00; same best policy yes (n=9) |
| tensorrt_fp16 | spearman +0.95; kendall +0.89; pairwise 0.94; same best policy yes (n=9) | spearman +1.00; kendall +1.00; pairwise 1.00; same best policy yes (n=9) |
| latent_noise_0.5 | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy no (n=9) | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy no (n=9) |
| latent_noise_0 | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) |
| bf16_autocast | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) |
| int8_weight_only | spearman +0.95; kendall +0.89; pairwise 0.94; same best policy yes (n=9) | spearman +1.00; kendall +1.00; pairwise 1.00; same best policy yes (n=9) |
| low_rank_0.5 | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) |
| low_rank_0.25 | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) |

Ranking, best first: simulator: actor > noise_0.1 > noise_0.3 > noise_0.2 > noise_0.4 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- baseline: actor > noise_0.1 > noise_0.2 > noise_0.4 > noise_0.3 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- baseline_reseed: actor > noise_0.1 > noise_0.2 > noise_0.3 > noise_0.4 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- gumbel: actor > noise_0.1 > noise_0.2 > noise_0.4 > noise_0.3 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- tensorrt_fp16: actor > noise_0.1 > noise_0.2 > noise_0.4 > noise_0.3 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- latent_noise_0.5: noise_0.1 > actor > noise_0.2 > noise_0.3 > noise_0.4 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- latent_noise_0: actor > noise_0.1 > noise_0.2 > noise_0.3 > noise_0.4 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- bf16_autocast: actor > noise_0.1 > noise_0.2 > noise_0.3 > noise_0.4 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- int8_weight_only: actor > noise_0.1 > noise_0.2 > noise_0.4 > noise_0.3 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- low_rank_0.5: actor > noise_0.1 > noise_0.2 > noise_0.3 > noise_0.4 > noise_0.5 > noise_0.6 > noise_0.8 > noise_1
- low_rank_0.25: actor > noise_0.1 > noise_0.2 > noise_0.3 > noise_0.4 > noise_0.5 > noise_0.6 > noise_1 > noise_0.8
