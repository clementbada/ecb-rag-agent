# Retrieval scorecard

Split: **dev** (20 questions with gold evidence). k = 5. Gold chunks = chunks covering at least 50% of a hand-verified evidence span, so variants with different chunk boundaries are scored on the same facts. Cost includes query embedding and rerank calls. MRR_ci95: 95% bootstrap interval over questions; configs whose intervals overlap are not separable on this set.

| variant              | mode          |   recall@5 |   precision@5 |   MRR | MRR_ci95     |   nDCG@5 |   doc_hit@5 |   latency_p50_ms |   cost_per_query_usd |
|:---------------------|:--------------|-----------:|--------------:|------:|:-------------|---------:|------------:|-----------------:|---------------------:|
| A_fixed200           | dense         |      0.412 |         0.110 | 0.243 | [0.12, 0.38] |    0.272 |       0.550 |          357.365 |                0.000 |
| A_fixed200           | bm25          |      0.613 |         0.160 | 0.403 | [0.24, 0.57] |    0.444 |       0.750 |           15.139 |                0.000 |
| A_fixed200           | hybrid        |      0.588 |         0.140 | 0.381 | [0.22, 0.55] |    0.417 |       0.750 |           17.664 |                0.000 |
| A_fixed200           | hybrid_rerank |      0.688 |         0.180 | 0.717 | [0.52, 0.90] |    0.690 |       0.800 |         1866.869 |                0.001 |
| B_fixed200_header    | dense         |      0.550 |         0.150 | 0.406 | [0.24, 0.58] |    0.403 |       0.700 |            0.451 |                0.000 |
| B_fixed200_header    | bm25          |      0.812 |         0.230 | 0.713 | [0.53, 0.87] |    0.701 |       0.900 |           18.561 |                0.000 |
| B_fixed200_header    | hybrid        |      0.787 |         0.220 | 0.658 | [0.48, 0.82] |    0.655 |       0.850 |           19.147 |                0.000 |
| B_fixed200_header    | hybrid_rerank |      0.912 |         0.260 | 0.912 | [0.78, 1.00] |    0.908 |       0.950 |         1864.454 |                0.001 |
| C_sentence250_header | dense         |      0.400 |         0.110 | 0.300 | [0.14, 0.47] |    0.302 |       0.550 |            0.432 |                0.000 |
| C_sentence250_header | bm25          |      0.900 |         0.230 | 0.685 | [0.52, 0.84] |    0.734 |       0.950 |           13.248 |                0.000 |
| C_sentence250_header | hybrid        |      0.750 |         0.200 | 0.575 | [0.41, 0.75] |    0.602 |       0.900 |           14.167 |                0.000 |
| C_sentence250_header | hybrid_rerank |      0.900 |         0.230 | 0.910 | [0.77, 1.00] |    0.893 |       0.950 |         1855.173 |                0.001 |
