Policy returns on the Dreamer walker (6 real episodes per policy; imagined over 60 steps from 24 held-out start states).

| policy | true return | baseline | baseline_reseed | gumbel | tensorrt_fp16 | latent_noise_0.5 | latent_noise_0 | bf16_autocast | int8_weight_only | low_rank_0.5 | low_rank_0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| actor | 784 | 111.7 | 111.8 | 111.7 | 112.6 | 105.6 | 99.0 | 111.8 | 113.2 | 100.1 | 40.2 |
| noise_0.2 | 751 | 108.6 | 106.6 | 108.6 | 108.9 | 105.3 | 94.4 | 109.8 | 109.3 | 92.3 | 35.6 |
| noise_0.5 | 600 | 96.1 | 100.0 | 96.1 | 94.3 | 91.0 | 74.0 | 96.8 | 94.2 | 71.4 | 31.6 |
| hold2 | 542 | 103.8 | 101.1 | 103.8 | 103.0 | 97.8 | 84.5 | 101.8 | 102.2 | 76.3 | 37.1 |
| noise_1.0 | 287 | 50.1 | 59.0 | 50.1 | 49.9 | 55.0 | 41.9 | 52.4 | 49.8 | 41.8 | 23.6 |
| half | 116 | 59.4 | 51.2 | 59.4 | 56.9 | 45.0 | 34.9 | 66.0 | 55.0 | 38.4 | 20.7 |
| random | 45 | 14.9 | 14.9 | 14.9 | 14.9 | 16.0 | 14.3 | 14.6 | 15.3 | 15.5 | 13.2 |
| inverted | 44 | 11.4 | 10.9 | 11.4 | 11.3 | 11.6 | 11.4 | 11.2 | 11.0 | 11.9 | 9.3 |
| zero | 20 | 15.4 | 14.0 | 15.4 | 15.4 | 15.5 | 13.5 | 15.8 | 16.0 | 15.7 | 12.1 |

Agreement of each model's ranking with the real simulator and with the unoptimized model's:

| model | vs simulator | vs unoptimized model |
|---|---|---|
| baseline | spearman +0.92; kendall +0.78; pairwise 0.89; same best policy yes (n=9) |  |
| baseline_reseed | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) |
| gumbel | spearman +0.92; kendall +0.78; pairwise 0.89; same best policy yes (n=9) | spearman +1.00; kendall +1.00; pairwise 1.00; same best policy yes (n=9) |
| tensorrt_fp16 | spearman +0.92; kendall +0.78; pairwise 0.89; same best policy yes (n=9) | spearman +1.00; kendall +1.00; pairwise 1.00; same best policy yes (n=9) |
| latent_noise_0.5 | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) |
| latent_noise_0 | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) | spearman +0.97; kendall +0.89; pairwise 0.94; same best policy yes (n=9) |
| bf16_autocast | spearman +0.92; kendall +0.78; pairwise 0.89; same best policy yes (n=9) | spearman +1.00; kendall +1.00; pairwise 1.00; same best policy yes (n=9) |
| int8_weight_only | spearman +0.92; kendall +0.78; pairwise 0.89; same best policy yes (n=9) | spearman +1.00; kendall +1.00; pairwise 1.00; same best policy yes (n=9) |
| low_rank_0.5 | spearman +0.93; kendall +0.83; pairwise 0.92; same best policy yes (n=9) | spearman +0.98; kendall +0.94; pairwise 0.97; same best policy yes (n=9) |
| low_rank_0.25 | spearman +0.93; kendall +0.83; pairwise 0.92; same best policy yes (n=9) | spearman +0.95; kendall +0.83; pairwise 0.92; same best policy yes (n=9) |

Ranking, best first: simulator: actor > noise_0.2 > noise_0.5 > hold2 > noise_1.0 > half > random > inverted > zero
- baseline: actor > noise_0.2 > hold2 > noise_0.5 > half > noise_1.0 > zero > random > inverted
- baseline_reseed: actor > noise_0.2 > hold2 > noise_0.5 > noise_1.0 > half > random > zero > inverted
- gumbel: actor > noise_0.2 > hold2 > noise_0.5 > half > noise_1.0 > zero > random > inverted
- tensorrt_fp16: actor > noise_0.2 > hold2 > noise_0.5 > half > noise_1.0 > zero > random > inverted
- latent_noise_0.5: actor > noise_0.2 > hold2 > noise_0.5 > noise_1.0 > half > random > zero > inverted
- latent_noise_0: actor > noise_0.2 > hold2 > noise_0.5 > noise_1.0 > half > random > zero > inverted
- bf16_autocast: actor > noise_0.2 > hold2 > noise_0.5 > half > noise_1.0 > zero > random > inverted
- int8_weight_only: actor > noise_0.2 > hold2 > noise_0.5 > half > noise_1.0 > zero > random > inverted
- low_rank_0.5: actor > noise_0.2 > hold2 > noise_0.5 > noise_1.0 > half > zero > random > inverted
- low_rank_0.25: actor > hold2 > noise_0.2 > noise_0.5 > noise_1.0 > half > random > zero > inverted
